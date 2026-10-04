# Roadmap: thesis pipeline

## Status 2026-09-30

Done:
- Event definition: inactivity > 180 days before 2025-12-31, otherwise censored.
- Leakage fixes: exit date and snapshot attributes are excluded (status flags, current pay grade, "ZZ_" values; see `tc.snapshot_leakage_columns`). All datasets are regenerated.
- 80/20 split with 5-fold CV and random-search tuning (`models/tuning_calibration.py`, `tuning.ipynb`). `advancedModels.ipynb` uses the same split and the tuned settings.
- Category experiments, Retention Index and permutation importance (`models/retention_index.py`). Outputs are in `models/ri_output/`, `models/benchmark_output/`, `models/tuning_output/` and `models/base_output/`.
- Thesis: restructured (Methods/Results/Discussion), new results and abstract.

Open:
- [x] `orgRef` (recruiting channel) is fixed at hiring (confirmed by the user), so it is not leakage. The model depends strongly on it: C-index 0.802 vs 0.685 without it.
- [x] lifelines CoxPH did not converge because some columns were constant in the training part. Fixed: CoxPH drops them per fold. All datasets now converge.
- [ ] Test split is only approximately stratified. `tc.holdout_split` uses StratifiedGroupKFold, and with one row per caregiver the test event rate deviates by up to 5.4 points; `StratifiedKFold` would be exact. Documented in the thesis, not re-run.
- [ ] `ausgesch-am` uses mixed date formats. The pipeline parses it with `%Y-%m-%d %H:%M:%S` and loses 1,334 exit dates; `format="mixed"` fixes this. The event status changes for only 2 caregivers. Documented in the thesis, not re-run.
- [ ] The assignment-level `selfOrganized` feature is numeric, so `createTotalJoin` does not aggregate it (only bool columns are). It is therefore missing from the model data.
- [ ] `combination/daysFromStart_modelData.csv` is obsolete (renamed to daysAfterStart). Delete it manually if you want.
- [ ] Optional: chronological validation, a sensitivity analysis for the 180-day grace period, and a performance refactor of the combination notebook (groupby instead of iterrows).

Review findings and open work for the caregiver-retention survival pipeline. `config.py` (URL, TOKEN, `pflege_map`, `driverMap`) and all CSVs are gitignored, so the findings below come from reading the code, not from running it on the data.

---

## 1. Pipeline overview

| Stage | Notebooks | What it does |
|---|---|---|
| Extraction | `extraction/caregiverWorkflow`, `clientWorkflow`, `assignmentWorkflow` | Pulls data from the REST API (paged, 2017–2025), flattens nested JSON, and drops PII through an exceptions list. |
| EDA / cleaning | `exploratory data analysis/caregiverAnalysis`, `clientAnalysis`, `assignmentAnalysis` | Drops columns with >10% missing values, then rows with remaining missing values. Removes near-constant columns (>99% one value) and one-hot encodes categoricals. Also groups `orgRef`, maps package grade to pay, splits `dateOfBirth`, merges the reviews into `PFKomp_mean`, and merges four booleans into `selfOrganized`. |
| Combination | `combination/combinationWorkflow` | `createTotalJoin` builds the caregiver-level features. `createTemporalCut` and `createTemporalFeatureDataset` create 7 cut types. Writes `totalData.csv` and `<cut>_modelData.csv`. `feature_groups` holds the category definitions. |
| Base models | `models/baseModels` | Kaplan-Meier, log-rank test, and univariate and multivariate CoxPH on `totalData.csv`. |
| Advanced models | `models/advancedModels` | `cross_validate_survival_models` runs 5-fold StratifiedGroupKFold over LinReg, sksurv Cox, lifelines CoxPH, RSF, GBS, Survival SVM and DeepSurv. Metrics are C-index, Brier, IBS, time-dependent AUC and binned KM calibration. |

Advanced-model details:
- Median imputation and scaling are fitted inside each fold.
- The tree models use `random_state=42`. RSF uses 300 trees.
- GBS uses the `coxph` loss, 200 estimators, learning rate 0.05 and depth 3.
- There is no hyperparameter tuning.

---

## 2. Gaps between the thesis and the code

- [ ] **The Retention Index is not implemented.** Missing: the overall index, the category-specific indices, the model selection, and the validation (calibration, discrimination, KM stratification).
- [ ] **Category-only and leave-one-category-out models don't exist.** `feature_groups` is only used for correlation summaries.
- [ ] **The feature groups don't match the thesis.** The code has demographics / experience / compensation / work_history / patient_care_requirements. The thesis has 4: Demographic & Skills, Monetary, Work History, Client Context.
- [ ] **There is no held-out 20% test set.** The thesis describes an 80/20 split plus 5-fold CV on the 80%. The current CV function uses all the data. Only the older `survival_model_benchmark` splits 80/20.
- [ ] **`advancedModels` reads `ratio_modelData.csv`**, which is the cut the thesis excludes for leakage. Other cuts can only be run by editing that one line.
- [ ] **The date cut is wrong.** The config value `"20210207000000"` parses as 7 Feb 2021. The thesis says 2 July 2021, the midpoint of the observation period.
- [ ] **Cut names differ.** The code says `daysFromStart`, the thesis says `daysAfterStart`.

---

## 3. Bugs and leakage

### baseModels
- [ ] **Missing comma in cell 3.** `"totalDays" "eventHasHappend"` concatenates into one string. This is harmless today only because the inner `leakage_columns` list catches both columns.
- [ ] **Cox and KM run on `totalData.csv`, not on a temporal cut.** `workRatio`, `averageRestLength` and `longestRest` are computed over the full history up to exit, so they encode the duration. The workRatio hazard ratio of about 141 partly reflects this.

### combinationWorkflow
- [ ] **Target leakage in `createTemporalFeatureDataset`.** It restores the original caregiver columns after computing features, so `ausgesch-am`, `eingestellt-am` and `endDate` end up in `*_modelData.csv`. Drop them or move them to a separate meta frame. Keep only `remainingDays` and `eventHasHappend` as targets.
- [ ] **Three event definitions disagree:**
  - `createTotalJoin` uses `endofCaregiver < cutoff`.
  - `createTemporalCut` uses `ausgesch-am` in (cut, 2025-12-31].
  - `createTemporalFeatureDataset` uses `targetEndDate` (from `getEndDate`, which falls back to the last assignment end) < observation_end and > cut. Caregivers without an exit date can therefore become events.

  Replace all three with one `is_event()` helper.
- [ ] **`cut()` can overwrite rows.** `df.loc[len(df)] = ...` runs on a filtered frame that keeps the original index labels, so a new row can land on an existing label. Collect the split parts and use `pd.concat(..., ignore_index=True)`.
- [ ] **The default `getCutDate` branch is broken.** `caregivers["ausgesch-am"] != None` is always True for `NaT`. Use `pd.notna()`.
- [ ] **The blanket `df.dropna()` (cell 11)** drops every caregiver missing any feature, such as patientAge or adminPacket stats. This causes selection bias. Report the drops per column and drop or impute selectively.
- [ ] **`from config import driverMap` fails.** There is no `config.py` in `combination/`. The rename is also not applied to the temporal datasets, so column names differ between `totalData.csv` and `*_modelData.csv`.
- [ ] **Patient-age else branches are dead code.** They assign unused locals. Set `caregiverRow[...] = np.nan` instead, as the adminPacket block does.
- [ ] **The `random` and `normal` cuts use no seed**, so every run produces a different `random_modelData.csv`.

### Possible status leakage in static caregiver columns
These columns describe the status at extraction time, and no leakage list excludes them. Check each one against the data:
- [ ] `Betreuer:innen_Austritt_Ja`
- [ ] `unpassend`
- [ ] `istangest`
- [ ] `isPaymentBlocked`
- [ ] `Gewerbe_ruhend_Ja`
- [ ] `aussch-grund` (if it survives the cleaning)

### Other
- [ ] The ratio-cut scores are suspiciously high. This is plausible leakage, because the cut position encodes the duration.

---

## 4. Tuning and calibration module (drafted, not in this repo yet)

This was drafted in a separate session. The two files still need to be added to `models/`:
- `tuning_calibration.py` holds the logic.
- `tuningAndCalibration.ipynb` is a thin runner where you pick the dataset and models.

**Tuning**
- Stratified 80/20 split, then random search with 5-fold StratifiedGroupKFold on the 80% only.
- Imputation and scaling are fitted inside each fold.
- The current defaults are always candidate 0, so you can see whether tuning helped.
- The objective is IBS for models that produce survival curves and C-index for the Survival SVM. On synthetic data, C-index-only tuning picked a worse-calibrated model (calibration error 0.072 vs 0.037).
- The Survival SVM is tuned on `alpha` only, with `rank_ratio=1.0`.

**Calibration** (judged on the untouched 20%)
- Calibration curves: predicted vs KM survival per decile at 30, 180 and 365 days.
- Cox-Snell Q-Q: residuals vs Exp(1), with censoring handled.
- PIT Q-Q: S(T|x) vs Uniform(0,1), with a D-calibration p-value.
- Output goes to `tuning_output/`, which is gitignored.

**Status and caveats**
- Tested only on 4,000 synthetic subjects with a Weibull stand-in model. Well-calibrated test model: D-cal p = 0.73. Deliberately miscalibrated model: p < 0.001.
- The sksurv and lifelines adapters (RSF, GBS, Cox, CoxPH, SVM) have never been executed. DeepSurv is not included.
- The metrics are custom numpy implementations and may differ from scikit-survival in the 3rd decimal. Keep the existing notebook for the thesis tables.
- Time-dependent AUC is not included.
- Calibration returns NaN for a decile when follow-up ends before the horizon with a censored observation. The existing notebook extrapolates there instead, so its `calibration_365` values will differ.
- The held-out set is small (about 20%), so the curves will be noisy on the smaller cuts.

---

## 5. Code quality: combinationWorkflow

- [ ] Add a constants cell for `OBS_START` (2017-01-01), `OBS_END` (2025-12-31), `COVID_START`/`COVID_END`, `SEED` and the input/output paths. Some of these dates are currently redefined in 5+ places.
- [ ] Move the functions into `combination/features.py` (`createTotalJoin`, `cut`, `createTemporalCut`, `createTemporalFeatureDataset`) and `combination/analysis.py` (`analyze_distributions`, `analyze_feature_relationships`, `summarize_correlations_by_group`).
- [ ] Move `feature_groups` next to `driverMap` in a shared config.
- [ ] Replace the per-caregiver `assignments[assignments["assignee"] == id]` filtering inside `iterrows` with a single `groupby("assignee")`.
- [ ] Add markdown headers: Load → Clean → Caregiver dataset → EDA → Temporal cuts → Export.
- [ ] Replace the nested `def getCutDate` inside `match` with a `{cutType: fn}` dict.
- [ ] Remove the duplicated `cutoff_date` and the redundant `to_datetime` calls. Normalize tabs to 4 spaces.

---

## 6. Methodology notes

- `analyze_feature_relationships` runs about 100 tests without correction. Apply Benjamini–Hochberg to the p-values.
- Correlating features with `totalDays` ignores censoring. Use a univariate Cox model or log-rank test per feature instead.
- `averageP` in the group summary is not meaningful. Report the number of significant features per group instead.

---

## 7. Next steps (in order)

1. [ ] Fix the small mismatches: the comma, the date cut (→ 2021-07-02), the cut names, and the 4 thesis feature groups.
2. [ ] Fix the combination leakage and event-definition bugs, then regenerate all `*_modelData.csv`.
3. [ ] Add the 80/20 split with CV on the training part, and run the leakage audit of the static columns.
4. [ ] Add the tuning and calibration module and run it on the real data.
5. [ ] Run the category-only and leave-one-category-out experiments.
6. [ ] Build the Retention Index, **RI = 100 · Ŝ(h | x)** at h = 30 / 180 / 365 days:
   - Use the tuned GBS or RSF on the fixed-date cut.
   - Add category-specific indices.
   - Validate with calibration, discrimination and KM stratification.
