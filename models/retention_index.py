"""
Modular feature-category analysis and Retention Index.

Builds on tuning_calibration.py (same data loading, 80/20 split,
preprocessing, metrics and calibration diagnostics).

1. Feature categories
   Every model feature is assigned to one of four conceptual categories
   (Demographic & Skills, Monetary, Work History & Intensity, Client
   Context) based on its origin, not on its predictive performance.

2. Modular analysis
   The selected model is trained on
     - all features,
     - the features of a single category (category-only),
     - all features except one category (leave-one-category-out),
   and evaluated with 5-fold CV on the training part and once on the
   held-out test set.

3. Retention Index
   RI_h(x)     = 100 * S(h | x)      overall model, all features
   RI_{h,k}(x) = 100 * S_k(h | x_k)  category-only model of category k
   for h = 30, 180, 365 days, calculated for the held-out test set and
   validated by calibration, discrimination and Kaplan-Meier
   stratification (RI quintiles).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import tuning_calibration as tc


HORIZONS = (30, 180, 365)


# =====================================================================
# FEATURE CATEGORIES
# =====================================================================

CATEGORIES = [
    "Demographic & Skills",
    "Monetary",
    "Work History & Intensity",
    "Client Context",
]

SHORT_NAMES = {
    "Demographic & Skills": "DEM",
    "Monetary": "MON",
    "Work History & Intensity": "WORK",
    "Client Context": "CLIENT",
}

# Features calculated from the caregiver's own assignment history
WORK_HISTORY_FEATURES = {
    "totalDays",
    "totalWorkDays",
    "totalRestDays",
    "workRatio",
    "averageAssignmentLength",
    "averageRestLength",
    "longestAssignment",
    "longestRest",
    "workedDuringCovid",
    "assignmentsBeforeCut",
    "isFirstAssignment_ratio",
}

# Compensation, billing and payment related features
MONETARY_FEATURES = {
    "pflegestatus-paket",
    "macht-stundenpak",
    "arrivalNoBilling",
    "departureNoBilling",
}

MONETARY_PREFIXES = (
    "adminPacket",
    "admin-betr-",
)


def feature_category(column):
    """Conceptual category of a model feature."""

    if column in WORK_HISTORY_FEATURES:
        return "Work History & Intensity"

    if column in MONETARY_FEATURES or column.startswith(MONETARY_PREFIXES):
        return "Monetary"

    # Duration-weighted client features and client age
    if column.endswith("_ratio") or column.startswith("patientAge"):
        return "Client Context"

    # Static caregiver characteristics: demographics, origin,
    # qualifications, experience, reviews, preferences
    return "Demographic & Skills"


def category_table(columns):
    """One row per feature with its category."""

    return pd.DataFrame({
        "feature": list(columns),
        "category": [feature_category(c) for c in columns],
    })


def category_columns(columns):
    """{category: [columns]}"""

    table = category_table(columns)
    return {
        category: table.loc[table["category"] == category, "feature"].tolist()
        for category in CATEGORIES
    }


# =====================================================================
# MODEL SETTINGS
# =====================================================================

def load_best_params(dataset_name, model_name, tuning_dir="tuning_output"):
    """Tuned parameters from tuning.ipynb, or the defaults."""

    path = Path(tuning_dir) / dataset_name / "best_params.json"

    if path.exists():
        params = json.loads(path.read_text()).get(model_name)
        if params:
            return params, "tuned"

    return dict(tc.MODEL_DEFAULTS[model_name]), "default"


def _fit_predict(model_name, params, X_train, y_train, X_test, random_state, n_jobs):

    model = tc.make_model(model_name, params, random_state=random_state, n_jobs=n_jobs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tc.fit_predict(model, X_train, y_train, X_test)


# =====================================================================
# MODULAR ANALYSIS
# =====================================================================

def feature_sets(columns):
    """All, category-only and leave-one-category-out feature sets."""

    groups = category_columns(columns)

    sets = [("All features", "all", list(columns))]

    for category in CATEGORIES:
        if groups[category]:
            sets.append((category, "only", groups[category]))

    for category in CATEGORIES:
        if groups[category]:
            remaining = [c for c in columns if c not in set(groups[category])]
            sets.append((f"All - {category}", "without", remaining))

    return sets


def _cv_fold(model_name, params, X, durations, events, train_idx, valid_idx, random_state):

    y_train = Surv.from_arrays(event=events[train_idx], time=durations[train_idx])

    risk, survival, times, _ = _fit_predict(
        model_name, params, X.iloc[train_idx], y_train, X.iloc[valid_idx], random_state, 1
    )

    return tc.evaluate(
        risk,
        survival,
        times,
        durations[train_idx],
        events[train_idx],
        durations[valid_idx],
        events[valid_idx],
        HORIZONS,
    )


def modular_analysis(
    dataset_path,
    model_name="GBS",
    n_splits=5,
    random_state=42,
    output_dir="ri_output",
    n_jobs=-1,
    verbose=True,
):
    """
    Category-only and leave-one-category-out experiments.

    Returns a table with one row per feature set: CV mean/std on the
    training part and the value on the held-out test set.
    """

    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem.replace("_modelData", "")
    output = Path(output_dir) / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    X, durations, events, groups = tc.load_model_data(dataset_path)
    train_idx, test_idx = tc.holdout_split(events, groups, random_state=random_state)

    params, setting = load_best_params(dataset_name, model_name)

    category_table(X.columns).to_csv(output / "feature_categories.csv", sep=";", index=False)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [
        (train_idx[a], train_idx[b])
        for a, b in cv.split(X.iloc[train_idx], events[train_idx], groups[train_idx])
    ]

    y_train = Surv.from_arrays(event=events[train_idx], time=durations[train_idx])

    rows = []

    for name, kind, columns in feature_sets(list(X.columns)):

        X_set = X[columns]

        fold_metrics = pd.DataFrame(
            Parallel(n_jobs=min(n_splits, n_jobs if n_jobs > 0 else n_splits))(
                delayed(_cv_fold)(
                    model_name, params, X_set, durations, events, a, b, random_state
                )
                for a, b in folds
            )
        )

        risk, survival, times, _ = _fit_predict(
            model_name, params, X_set.iloc[train_idx], y_train, X_set.iloc[test_idx],
            random_state, -1,
        )

        test_metrics = tc.evaluate(
            risk, survival, times,
            durations[train_idx], events[train_idx],
            durations[test_idx], events[test_idx],
            HORIZONS,
        )

        for horizon in HORIZONS:
            table = tc.calibration_table(
                survival, times, durations[test_idx], events[test_idx], horizon
            )
            test_metrics[f"calibration_{horizon}"] = table["absolute_error"].mean()

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "setting": setting,
            "feature_set": name,
            "kind": kind,
            "n_features": len(columns),
        }
        for column in fold_metrics.columns:
            row[f"{column}_cv_mean"] = fold_metrics[column].mean()
            row[f"{column}_cv_std"] = fold_metrics[column].std()
        for key, value in test_metrics.items():
            row[f"{key}_test"] = value

        rows.append(row)

        if verbose:
            print(
                f"[{dataset_name}] {name:38} p={len(columns):3d} "
                f"C_cv={row['c_index_cv_mean']:.3f} IBS_cv={row['integrated_brier_score_cv_mean']:.3f} "
                f"C_test={row['c_index_test']:.3f} IBS_test={row['integrated_brier_score_test']:.3f}"
            )

    results = pd.DataFrame(rows)

    # Differences to the complete model (leave-one-category-out)
    full = results.loc[results["kind"] == "all"].iloc[0]
    for metric in ["c_index", "integrated_brier_score", "auc_365"]:
        for part in ["cv_mean", "test"]:
            column = f"{metric}_{part}"
            results[f"delta_{column}"] = results[column] - full[column]

    results.to_csv(output / f"modular_{model_name}.csv", sep=";", index=False)

    return results


# =====================================================================
# RETENTION INDEX
# =====================================================================

def retention_index(
    dataset_path,
    model_name="GBS",
    random_state=42,
    output_dir="ri_output",
    n_groups=5,
    show_plots=True,
):
    """
    Overall and category-specific Retention Index on the held-out test
    set, with validation.

    Writes to <output_dir>/<dataset>/:
        retention_index_test.csv     RI per test caregiver
        ri_validation.csv            discrimination / calibration per index
        ri_stratification.csv       KM survival per RI365 quintile
        ri_examples.csv              representative caregivers
        ri_distribution.png, ri_stratification.png,
        ri_calibration_<index>.png

    Returns a dict with the tables.
    """

    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem.replace("_modelData", "")
    output = Path(output_dir) / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    X, durations, events, groups = tc.load_model_data(dataset_path)
    train_idx, test_idx = tc.holdout_split(events, groups, random_state=random_state)

    params, setting = load_best_params(dataset_name, model_name)

    y_train = Surv.from_arrays(event=events[train_idx], time=durations[train_idx])
    d_train, e_train = durations[train_idx], events[train_idx]
    d_test, e_test = durations[test_idx], events[test_idx]

    index_sets = [("Overall", list(X.columns))]
    for category, columns in category_columns(list(X.columns)).items():
        if columns:
            index_sets.append((category, columns))

    ri = pd.DataFrame({
        "id": groups[test_idx],
        "duration": d_test,
        "event": e_test,
    })

    validation_rows = []

    for name, columns in index_sets:

        risk, survival, times, _ = _fit_predict(
            model_name, params,
            X[columns].iloc[train_idx], y_train, X[columns].iloc[test_idx],
            random_state, -1,
        )

        label = "Overall" if name == "Overall" else SHORT_NAMES[name]

        for horizon in HORIZONS:
            ri[f"RI{horizon}_{label}"] = 100 * tc.survival_at(survival, times, horizon)

        metrics = tc.evaluate(risk, survival, times, d_train, e_train, d_test, e_test, HORIZONS)

        # Discrimination of the index itself: low RI365 = high risk
        metrics["c_index_RI365"] = concordance_index_censored(
            e_test, d_test, -ri[f"RI365_{label}"].to_numpy()
        )[0]

        table, p_value = tc.plot_calibration(
            survival, times, d_test, e_test,
            title=f"{dataset_name} | Retention Index ({name}) | {model_name} ({setting}) | test set",
            horizons=HORIZONS,
            save_path=output / f"ri_calibration_{label}.png",
            show=show_plots and name == "Overall",
        )
        table.insert(0, "index", name)
        table.to_csv(output / f"ri_calibration_{label}.csv", sep=";", index=False)

        for horizon in HORIZONS:
            metrics[f"calibration_{horizon}"] = (
                table.loc[table["horizon"] == horizon, "absolute_error"].mean()
            )
            metrics[f"mean_RI{horizon}"] = ri[f"RI{horizon}_{label}"].mean()
        metrics["d_calibration_p"] = p_value

        validation_rows.append({"index": name, "n_features": len(columns), **metrics})

    validation = pd.DataFrame(validation_rows)

    # Observed KM survival of the whole test set, for reference
    km_all = KaplanMeierFitter().fit(d_test, e_test)
    validation.attrs["observed_S"] = {h: float(km_all.predict(h)) for h in HORIZONS}

    stratification, logrank = _stratification_plot(
        ri, "RI365_Overall", n_groups,
        title=f"{dataset_name} | Kaplan-Meier survival by overall RI365 quintile (test set)",
        save_path=output / "ri_stratification.png",
        show=show_plots,
    )

    _distribution_plot(
        ri,
        title=f"{dataset_name} | Overall Retention Index on the test set",
        save_path=output / "ri_distribution.png",
        show=show_plots,
    )

    examples = _representative_examples(ri)

    ri.to_csv(output / "retention_index_test.csv", sep=";", index=False)
    validation.to_csv(output / "ri_validation.csv", sep=";", index=False)
    stratification.to_csv(output / "ri_stratification.csv", sep=";", index=False)
    examples.to_csv(output / "ri_examples.csv", sep=";", index=False)

    with open(output / "ri_summary.json", "w", encoding="utf-8") as file:
        json.dump({
            "dataset": dataset_name,
            "model": model_name,
            "setting": setting,
            "params": params,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_event_rate": float(e_test.mean()),
            "observed_S_test": validation.attrs["observed_S"],
            "logrank_statistic": float(logrank.test_statistic),
            "logrank_p": float(logrank.p_value),
        }, file, indent=2)

    return {
        "ri": ri,
        "validation": validation,
        "stratification": stratification,
        "logrank": logrank,
        "examples": examples,
    }


def _stratification_plot(ri, column, n_groups, title, save_path=None, show=True):
    """KM curves per RI quantile group and a multivariate log-rank test."""

    data = ri.copy()
    data["group"] = pd.qcut(data[column], q=n_groups, labels=False, duplicates="drop") + 1

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rows = []

    for group, part in data.groupby("group"):

        km = KaplanMeierFitter().fit(
            part["duration"], part["event"],
            label=f"Q{group}: RI365 {part[column].min():.0f}-{part[column].max():.0f}",
        )
        km.plot_survival_function(ax=ax, ci_show=False)

        row = {
            "group": f"Q{group}",
            "n": len(part),
            "events": int(part["event"].sum()),
            "RI365_min": part[column].min(),
            "RI365_max": part[column].max(),
            "RI365_mean": part[column].mean(),
        }
        for horizon in HORIZONS:
            row[f"observed_S{horizon}"] = (
                float(km.predict(horizon)) if part["duration"].max() >= horizon else np.nan
            )
        rows.append(row)

    logrank = multivariate_logrank_test(data["duration"], data["group"], data["event"])

    ax.set_xlim(0, max(730, 0))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days after prediction date")
    ax.set_ylabel("Observed share still active (Kaplan-Meier)")
    ax.set_title(f"{title}\nlog-rank p = {logrank.p_value:.2g}", fontsize=11)
    for horizon in HORIZONS:
        ax.axvline(horizon, color="grey", linewidth=0.6, linestyle=":")
    ax.grid(alpha=0.2)
    ax.legend(title="Quintile (Q1 = lowest RI365)", loc="lower left", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return pd.DataFrame(rows), logrank


def _distribution_plot(ri, title, save_path=None, show=True):

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5 * len(HORIZONS), 4), sharey=True)

    for ax, horizon in zip(axes, HORIZONS):
        ax.hist(ri[f"RI{horizon}_Overall"], bins=20, range=(0, 100), edgecolor="black")
        ax.set_title(f"RI{horizon} (mean {ri[f'RI{horizon}_Overall'].mean():.1f})")
        ax.set_xlabel("Retention Index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Caregivers in test set")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _representative_examples(ri, quantiles=(0.1, 0.5, 0.9)):
    """Caregivers closest to fixed quantiles of the overall RI365."""

    rows = []
    for label, q in zip("ABC", quantiles):
        target = ri["RI365_Overall"].quantile(q)
        index = (ri["RI365_Overall"] - target).abs().idxmin()
        row = ri.loc[index].to_dict()
        row["example"] = label
        row["quantile"] = q
        rows.append(row)

    columns = ["example", "quantile", "duration", "event"] + [
        c for c in ri.columns if c.startswith("RI")
    ]
    return pd.DataFrame(rows)[columns]


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

def permutation_importance(
    dataset_path,
    model_name="GBS",
    n_repeats=10,
    random_state=42,
    output_dir="ri_output",
):
    """
    Permutation importance on the held-out test set, for single
    features and for whole categories (all features of a category
    permuted together). Importance = drop in test C-index.
    """

    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem.replace("_modelData", "")
    output = Path(output_dir) / dataset_name
    output.mkdir(parents=True, exist_ok=True)

    X, durations, events, groups = tc.load_model_data(dataset_path)
    train_idx, test_idx = tc.holdout_split(events, groups, random_state=random_state)
    params, _ = load_best_params(dataset_name, model_name)

    y_train = Surv.from_arrays(event=events[train_idx], time=durations[train_idx])
    X_test = X.iloc[test_idx].reset_index(drop=True)
    d_test, e_test = durations[test_idx], events[test_idx]

    _, _, _, pipeline = _fit_predict(
        model_name, params, X.iloc[train_idx], y_train, X_test, random_state, -1
    )

    def c_index(frame):
        return concordance_index_censored(e_test, d_test, pipeline.predict(frame))[0]

    baseline = c_index(X_test)
    rng = np.random.default_rng(random_state)

    def importance(columns):
        drops = []
        for _ in range(n_repeats):
            permuted = X_test.copy()
            order = rng.permutation(len(permuted))
            permuted[columns] = permuted[columns].to_numpy()[order]
            drops.append(baseline - c_index(permuted))
        return np.mean(drops), np.std(drops)

    rows = []
    for column in X.columns:
        mean, std = importance([column])
        rows.append({
            "feature": column,
            "category": feature_category(column),
            "importance_mean": mean,
            "importance_std": std,
        })

    features = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)

    rows = []
    for category, columns in category_columns(list(X.columns)).items():
        if columns:
            mean, std = importance(columns)
            rows.append({
                "category": category,
                "n_features": len(columns),
                "importance_mean": mean,
                "importance_std": std,
            })

    categories = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)

    features.to_csv(output / f"permutation_importance_{model_name}.csv", sep=";", index=False)
    categories.to_csv(output / f"category_importance_{model_name}.csv", sep=";", index=False)

    return baseline, features, categories
