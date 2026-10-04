"""
Hyperparameter tuning and calibration for the survival models.

Workflow (per dataset):
    1. Stratified, grouped 80/20 split into a training and a held-out test set.
    2. Random search on the 80 % training part with 5-fold
       StratifiedGroupKFold. Imputation and scaling are fitted inside
       every fold. Candidate 0 is always the current default setting,
       so the output shows whether tuning helps at all.
    3. The best setting (lowest mean Integrated Brier Score) is refitted
       on the full training part and evaluated once on the untouched
       test set, next to the default setting.
    4. Calibration on the test set: decile calibration curves at fixed
       horizons, Cox-Snell residual plot and a PIT / D-calibration plot.

The metrics come from scikit-survival, the same functions as in
advancedModels.ipynb, so the numbers are comparable.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import loguniform, randint, uniform

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter, NelsonAalenFitter

from sksurv.util import Surv
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest


# =====================================================================
# DATA
# =====================================================================

# Same exclusions as cross_validate_survival_models in advancedModels.ipynb
DEFAULT_EXCLUSIONS = [
    "id",
    "remainingDays",
    "eventHasHappend",
    "targetEndDate",
    "assignmentsAfterCut",
    "ausgesch-am",
    "cutDate",
    "startofCaregiver",
    "endofCaregiver",
    "eingestellt-am",
]

# Caregiver attributes that describe the state at extraction time, not at
# the prediction point, and can therefore leak the outcome:
#   - status flags that are set when a caregiver leaves
#   - the current pay grade: caregivers who left before a grade
#     migration keep a discontinued "ZZ_" grade (100 % events)
STATUS_COLUMNS = [
    "istangest",
    "unpassend",
    "isPaymentBlocked",
    "pflegestatus-paket",
]

# One-hot columns of discontinued ("ZZ_") values, e.g. transport
# companies that only former caregivers still have assigned
DISCONTINUED_MARKER = "_ZZ_"


def snapshot_leakage_columns(columns):
    """Columns excluded because they describe the state at extraction."""

    return [
        column for column in columns
        if column in STATUS_COLUMNS or DISCONTINUED_MARKER in column
    ]


def load_model_data(
    path,
    duration_col="remainingDays",
    event_col="eventHasHappend",
    group_col="id",
    extra_exclude=None,
):
    """
    Read a <cut>_modelData.csv and return features, target and groups.

    Only numeric / boolean columns are used as features. Rows with a
    missing or non-positive duration are dropped. Columns that are
    all-missing or constant are removed.
    """

    data = pd.read_csv(path, sep=";", low_memory=False)

    data[duration_col] = pd.to_numeric(data[duration_col], errors="coerce")
    data = data[data[duration_col].notna() & (data[duration_col] > 0)].copy()

    exclusions = (
        set(DEFAULT_EXCLUSIONS)
        | set(snapshot_leakage_columns(data.columns))
        | {duration_col, event_col, group_col}
    )
    if extra_exclude is not None:
        exclusions |= set(extra_exclude)

    X = (
        data
        .select_dtypes(include=["number", "bool"])
        .drop(columns=[c for c in exclusions if c in data.columns], errors="ignore")
        .astype(float)
    )

    X = X.loc[:, ~X.isna().all()]
    X = X.loc[:, X.nunique(dropna=True) > 1]

    durations = data[duration_col].to_numpy(dtype=float)
    events = data[event_col].astype(bool).to_numpy()
    groups = data[group_col].to_numpy()

    return X.reset_index(drop=True), durations, events, groups


def holdout_split(events, groups, test_size=0.2, random_state=42):
    """
    Stratified (by event) and grouped (by caregiver id) train/test split.

    Uses the first fold of a StratifiedGroupKFold with round(1/test_size)
    splits, so the test set is ~test_size of the data and no caregiver
    appears in both parts.
    """

    n_splits = int(round(1 / test_size))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(np.zeros(len(events)), events, groups)
    )

    return train_idx, test_idx


# =====================================================================
# MODELS AND SEARCH SPACES
# =====================================================================

# Candidate 0 = the settings currently used in advancedModels.ipynb
MODEL_DEFAULTS = {
    "Cox": {
        "alpha": 0.01,
    },
    "RSF": {
        "n_estimators": 300,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "max_depth": None,
    },
    "GBS": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 1.0,
        "min_samples_leaf": 1,
        "max_features": None,
    },
}

SEARCH_SPACES = {
    # sksurv's ridge penalty acts on the summed partial likelihood,
    # so useful values scale with the sample size.
    "Cox": {
        "alpha": loguniform(1e-2, 1e4),
    },
    "RSF": {
        "n_estimators": [200, 300, 500],
        "min_samples_split": randint(4, 41),
        "min_samples_leaf": randint(3, 31),
        "max_features": ["sqrt", 0.1, 0.2, 0.33],
        "max_depth": [None, 8, 12, 20],
    },
    "GBS": {
        "n_estimators": randint(100, 601),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": [2, 3, 4, 5],
        "subsample": uniform(0.6, 0.4),
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [None, "sqrt", 0.3],
    },
}


def make_model(model_name, params, random_state=42, n_jobs=1):
    """Build an unfitted sksurv model from a parameter dict."""

    params = dict(params)

    if model_name == "Cox":
        return CoxPHSurvivalAnalysis(**params)

    if model_name == "RSF":
        return RandomSurvivalForest(
            **params,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_name == "GBS":
        return GradientBoostingSurvivalAnalysis(
            loss="coxph",
            **params,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model: {model_name}")


def sample_candidates(model_name, n_candidates, random_state=42):
    """Default setting first, then n_candidates - 1 random settings."""

    candidates = [dict(MODEL_DEFAULTS[model_name])]

    if n_candidates > 1:
        sampled = ParameterSampler(
            SEARCH_SPACES[model_name],
            n_iter=n_candidates - 1,
            random_state=random_state,
        )
        for params in sampled:
            candidates.append({
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in params.items()
            })

    return candidates


# =====================================================================
# PREDICTION HELPERS
# =====================================================================

def fit_predict(model, X_train, y_train, X_test):
    """
    Fit imputer + scaler + model on the training data and return
    (risk scores, survival matrix, time grid) for the test data.

    The survival matrix has one row per test subject, evaluated on the
    model's own event-time grid.
    """

    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        model,
    )

    pipeline.fit(X_train, y_train)

    risk = pipeline.predict(X_test)

    survival = pipeline.predict_survival_function(X_test, return_array=True)
    times = pipeline[-1].unique_times_

    return risk, np.asarray(survival), np.asarray(times, dtype=float), pipeline


def survival_at(survival, times, t):
    """
    Evaluate right-continuous step survival curves at time(s) t.

    t is a scalar (same time for every subject) or an array with one
    time per subject. Before the first grid time S = 1.
    """

    t = np.asarray(t, dtype=float)
    index = np.searchsorted(times, t, side="right") - 1

    if t.ndim == 0:
        if index < 0:
            return np.ones(survival.shape[0])
        return survival[:, index]

    result = np.ones(len(t))
    valid = index >= 0
    result[valid] = survival[np.flatnonzero(valid), index[valid]]
    return result


def ibs_time_grid(duration_train, duration_test, n_points=100):
    """Same IBS time range as advancedModels.ipynb (5th-90th percentile)."""

    lower = max(1, np.percentile(duration_test, 5))
    upper = min(
        np.percentile(duration_test, 90),
        duration_train.max() - 1,
        duration_test.max() - 1,
    )

    if upper <= lower:
        return None

    return np.linspace(lower, upper, n_points)


# =====================================================================
# METRICS
# =====================================================================

def evaluate(
    risk,
    survival,
    times,
    duration_train,
    event_train,
    duration_test,
    event_test,
    horizons=(30, 180, 365),
):
    """C-index, IBS, Brier score and time-dependent AUC per horizon."""

    y_train = Surv.from_arrays(event=event_train, time=duration_train)
    y_test = Surv.from_arrays(event=event_test, time=duration_test)

    result = {
        "c_index": concordance_index_censored(event_test, duration_test, risk)[0],
        "integrated_brier_score": np.nan,
    }

    grid = ibs_time_grid(duration_train, duration_test)
    if grid is not None:
        matrix = np.column_stack([survival_at(survival, times, t) for t in grid])
        result["integrated_brier_score"] = integrated_brier_score(
            y_train, y_test, matrix, grid
        )

    for horizon in horizons:
        result[f"brier_{horizon}"] = np.nan
        result[f"auc_{horizon}"] = np.nan

        if not (horizon < duration_train.max() and horizon < duration_test.max()):
            continue

        s_h = survival_at(survival, times, horizon)

        try:
            _, bs = brier_score(y_train, y_test, s_h.reshape(-1, 1), [horizon])
            result[f"brier_{horizon}"] = bs[0]
        except ValueError:
            pass

        try:
            auc, _ = cumulative_dynamic_auc(y_train, y_test, 1 - s_h, [horizon])
            result[f"auc_{horizon}"] = auc[0]
        except ValueError:
            pass

    return result


# =====================================================================
# RANDOM SEARCH
# =====================================================================

def _run_fold(model_name, params, X, durations, events, train_idx, test_idx,
              random_state, n_jobs):

    model = make_model(model_name, params, random_state=random_state, n_jobs=n_jobs)

    y_train = Surv.from_arrays(event=events[train_idx], time=durations[train_idx])

    start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        risk, survival, times, _ = fit_predict(
            model, X.iloc[train_idx], y_train, X.iloc[test_idx]
        )

    metrics = evaluate(
        risk,
        survival,
        times,
        durations[train_idx],
        events[train_idx],
        durations[test_idx],
        events[test_idx],
    )
    metrics["fit_seconds"] = time.time() - start

    return metrics


def random_search(
    model_name,
    X,
    durations,
    events,
    groups,
    n_candidates=20,
    n_splits=5,
    objective="integrated_brier_score",
    random_state=42,
    n_jobs=-1,
    results_path=None,
    verbose=True,
):
    """
    Random search with StratifiedGroupKFold on the given (training) data.

    Folds of one candidate run in parallel. The results table is written
    to results_path after every candidate, so an interrupted run keeps
    its progress.

    Returns the results table (one row per candidate, mean and std of
    every metric) sorted by the objective.
    """

    lower_is_better = objective == "integrated_brier_score" or objective.startswith("brier")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(X, events, groups))

    n_cpus = Parallel(n_jobs=n_jobs)._effective_n_jobs()
    fold_workers = min(n_splits, n_cpus)

    # RSF parallelises over trees itself; give each fold a share of the CPUs
    inner_jobs = max(1, n_cpus // fold_workers) if model_name == "RSF" else 1

    candidates = sample_candidates(model_name, n_candidates, random_state)
    rows = []

    for number, params in enumerate(candidates):

        start = time.time()

        fold_metrics = Parallel(n_jobs=fold_workers)(
            delayed(_run_fold)(
                model_name, params, X, durations, events,
                train_idx, test_idx, random_state, inner_jobs,
            )
            for train_idx, test_idx in folds
        )

        fold_metrics = pd.DataFrame(fold_metrics)

        row = {
            "model": model_name,
            "candidate": number,
            "is_default": number == 0,
            "params": json.dumps(params),
        }
        for column in fold_metrics.columns:
            row[f"{column}_mean"] = fold_metrics[column].mean()
            row[f"{column}_std"] = fold_metrics[column].std()

        rows.append(row)

        if verbose:
            print(
                f"[{model_name}] candidate {number + 1}/{len(candidates)} "
                f"IBS={row['integrated_brier_score_mean']:.4f} "
                f"C={row['c_index_mean']:.4f} "
                f"({time.time() - start:.0f}s) {params}"
            )

        results = pd.DataFrame(rows).sort_values(
            f"{objective}_mean", ascending=lower_is_better
        )

        if results_path is not None:
            results.to_csv(results_path, sep=";", index=False)

    return results.reset_index(drop=True)


# =====================================================================
# CALIBRATION
# =====================================================================

def calibration_table(survival, times, durations, events, horizon, bins=10):
    """
    Predicted vs. Kaplan-Meier observed survival at `horizon`, per
    decile of predicted survival.

    If a decile's follow-up ends before the horizon, the KM estimate
    would be an extrapolation; observed survival is NaN there.
    """

    predicted = survival_at(survival, times, horizon)

    table = pd.DataFrame({
        "predicted": predicted,
        "duration": durations,
        "event": events,
    })
    table["bin"] = pd.qcut(table["predicted"], q=bins, labels=False, duplicates="drop")

    rows = []
    for bin_id, group in table.groupby("bin"):

        observed = np.nan
        if group["duration"].max() >= horizon:
            km = KaplanMeierFitter().fit(group["duration"], group["event"])
            observed = float(km.predict(horizon))

        rows.append({
            "horizon": horizon,
            "bin": int(bin_id),
            "n": len(group),
            "predicted_survival": group["predicted"].mean(),
            "observed_survival": observed,
        })

    rows = pd.DataFrame(rows)
    rows["absolute_error"] = (rows["predicted_survival"] - rows["observed_survival"]).abs()
    return rows


def d_calibration(survival, times, durations, events, bins=10):
    """
    D-calibration (Haider et al., 2020).

    For a well-calibrated model S(T_i | x_i) is Uniform(0, 1).
    Events count fully in the bin that contains S(T_i). Censored
    subjects are spread uniformly over all values below S(C_i).

    Returns (bin proportions, chi-square statistic, p-value).
    """

    s = np.clip(survival_at(survival, times, durations), 1e-12, 1.0)
    edges = np.linspace(0, 1, bins + 1)
    counts = np.zeros(bins)

    bin_of_s = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)

    for value, k, event in zip(s, bin_of_s, events):

        if event:
            counts[k] += 1
            continue

        counts[k] += (value - edges[k]) / value
        counts[:k] += (edges[1] - edges[0]) / value

    expected = len(s) / bins
    statistic = ((counts - expected) ** 2 / expected).sum()
    p_value = stats.chi2.sf(statistic, df=bins - 1)

    return counts / counts.sum(), statistic, p_value


def cox_snell_residuals(survival, times, durations):
    """r_i = -log S(T_i | x_i). Censored Exp(1) if the model is calibrated."""

    s = np.clip(survival_at(survival, times, durations), 1e-12, 1.0)
    return -np.log(s)


def plot_calibration(
    survival,
    times,
    durations,
    events,
    title,
    horizons=(30, 180, 365),
    save_path=None,
    show=True,
):
    """
    One figure with a calibration curve per horizon, the Cox-Snell
    residual plot and the PIT / D-calibration plot.

    Returns (calibration table for all horizons, D-calibration p-value).
    """

    fig, axes = plt.subplots(1, len(horizons) + 2, figsize=(5 * (len(horizons) + 2), 4.8))

    tables = []

    # -- calibration curves ------------------------------------------
    for ax, horizon in zip(axes, horizons):

        table = calibration_table(survival, times, durations, events, horizon)
        tables.append(table)

        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
        ax.plot(
            table["predicted_survival"],
            table["observed_survival"],
            marker="o",
            label="Deciles",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted S({horizon})")
        ax.set_ylabel(f"Observed S({horizon}) (Kaplan-Meier)")
        ax.set_title(
            f"{horizon} days | mean abs. error "
            f"{table['absolute_error'].mean():.3f}"
        )
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)

    # -- Cox-Snell ----------------------------------------------------
    residuals = cox_snell_residuals(survival, times, durations)
    nelson_aalen = NelsonAalenFitter().fit(residuals, events)
    cumulative = nelson_aalen.cumulative_hazard_

    ax = axes[len(horizons)]
    upper = np.quantile(residuals, 0.99)
    ax.plot([0, upper], [0, upper], linestyle="--", color="grey", label="Exp(1)")
    ax.step(cumulative.index, cumulative.iloc[:, 0], where="post", label="Nelson-Aalen")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Cox-Snell residual")
    ax.set_ylabel("Cumulative hazard of residuals")
    ax.set_title("Cox-Snell residuals")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    # -- PIT / D-calibration -----------------------------------------
    proportions, _, p_value = d_calibration(survival, times, durations, events)

    ax = axes[len(horizons) + 1]
    cumulative_pit = np.concatenate([[0], np.cumsum(proportions)])
    grid = np.linspace(0, 1, len(proportions) + 1)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Uniform(0, 1)")
    ax.plot(grid, cumulative_pit, marker="o", label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uniform quantile")
    ax.set_ylabel("Quantile of S(T | x)")
    ax.set_title(f"PIT Q-Q | D-calibration p = {p_value:.3f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pd.concat(tables, ignore_index=True), p_value


# =====================================================================
# FULL WORKFLOW
# =====================================================================

def tune_and_evaluate(
    dataset_path,
    models=("GBS", "RSF", "Cox"),
    n_candidates=20,
    n_splits=5,
    test_size=0.2,
    horizons=(30, 180, 365),
    output_dir="tuning_output",
    extra_exclude=None,
    random_state=42,
    show_plots=True,
):
    """
    Run the whole workflow for one dataset.

    Writes to <output_dir>/<dataset name>/:
        holdout_ids.csv              ids of the held-out test set
        search_<model>.csv           random-search results
        best_params.json             best setting per model
        test_metrics.csv             default vs tuned on the test set
        calibration_<model>.csv      decile calibration tables
        calibration_<model>.png      calibration figure

    Returns the test metrics table.
    """

    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem.replace("_modelData", "")

    output = Path(output_dir) / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    X, durations, events, groups = load_model_data(dataset_path, extra_exclude=extra_exclude)
    train_idx, test_idx = holdout_split(events, groups, test_size, random_state)

    print(
        f"\n=== {dataset_name}: {len(X)} rows, {X.shape[1]} features, "
        f"event rate {events.mean():.1%} | train {len(train_idx)}, test {len(test_idx)}"
    )

    pd.DataFrame({"id": groups[test_idx]}).to_csv(
        output / "holdout_ids.csv", sep=";", index=False
    )

    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    d_train, d_test = durations[train_idx], durations[test_idx]
    e_train, e_test = events[train_idx], events[test_idx]
    g_train = groups[train_idx]

    y_train = Surv.from_arrays(event=e_train, time=d_train)

    best_params = {}
    test_rows = []

    for model_name in models:

        search = random_search(
            model_name,
            X_train,
            d_train,
            e_train,
            g_train,
            n_candidates=n_candidates,
            n_splits=n_splits,
            random_state=random_state,
            results_path=output / f"search_{model_name}.csv",
        )

        tuned = json.loads(search.iloc[0]["params"])
        best_params[model_name] = tuned

        settings = {"default": MODEL_DEFAULTS[model_name], "tuned": tuned}

        for setting, params in settings.items():

            model = make_model(model_name, params, random_state=random_state, n_jobs=-1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                risk, survival, times, _ = fit_predict(model, X_train, y_train, X_test)

            metrics = evaluate(
                risk, survival, times, d_train, e_train, d_test, e_test, horizons
            )

            table, p_value = plot_calibration(
                survival,
                times,
                d_test,
                e_test,
                title=f"{dataset_name} | {model_name} ({setting}) | held-out test set",
                horizons=horizons,
                save_path=output / f"calibration_{model_name}_{setting}.png",
                show=show_plots and setting == "tuned",
            )
            table.insert(0, "setting", setting)
            table.insert(0, "model", model_name)
            table.to_csv(output / f"calibration_{model_name}_{setting}.csv", sep=";", index=False)

            for horizon in horizons:
                metrics[f"calibration_{horizon}"] = (
                    table.loc[table["horizon"] == horizon, "absolute_error"].mean()
                )
            metrics["d_calibration_p"] = p_value

            test_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "setting": setting,
                **metrics,
                "params": json.dumps(params),
            })

    with open(output / "best_params.json", "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(output / "test_metrics.csv", sep=";", index=False)

    return test_metrics
