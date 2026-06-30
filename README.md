# FS-Pipeline: Feature Selection + Regression Model Benchmarking

A parameterized command-line pipeline that:

1. **Selects features** from a regression dataset using 11 different feature
   selection (FS) methods, at multiple feature-set sizes (e.g. top-100,
   top-200, top-400 ...).
2. **Trains and evaluates regression models** (11 algorithms) on every
   resulting (method × size) feature set, in parallel, and reports
   MAE / RMSE / Pearson correlation / R² on 5-fold CV and on a held-out
   validation set.

It was converted from an exploratory Jupyter notebook into a reusable,
fully argument-driven CLI tool, with several correctness bugs found in the
original notebook fixed along the way (see [Bug fixes](#bug-fixes-from-the-original-notebook)).

## Contents

| File | Purpose |
|---|---|
| `feature_selection_pipeline.py` | Runs all FS methods, saves per-method/per-size train & test CSVs + a combined JSON summary. |
| `auto_new1_scaler.py` | Trains + evaluates one regression algorithm on one train/test CSV pair. |
| `Auto_bash.sh` | Parallel orchestrator: runs `auto_new1_scaler.py` once per algorithm listed in `list`. |
| `run_pipeline.py` | End-to-end driver: feature selection, then training, across every method/size combination. |
| `list` | Example algorithm list file (one algorithm code per line) consumed by `Auto_bash.sh`. |
| `requirements.txt` | Python dependencies (core + optional). |

## Installation

```bash
git clone <this-repo-url>
cd fs-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`) are required.
A few FS methods depend on optional packages — if one isn't installed, the
pipeline prints a warning and skips just that method rather than crashing:

| Method | Package |
|---|---|
| `lgbm` | `lightgbm` |
| `relieff` | `skrebate` |
| `mrmr` | `mrmr_selection` (imported as `mrmr`) |
| `boruta` | `Boruta` |

Model training additionally supports `XGB` (XGBoost) if `xgboost` is
installed; it's skipped automatically (with a warning) when running `-m ALL`
if missing, and raises a clear error if `-m XGB` is explicitly requested
without it.

## Input data format

Both the train and test/validation CSV must have feature columns first,
followed by a block of label columns at the end:

```
feat_1, feat_2, ..., feat_N, lab_other_1, lab_other_2, ..., lab_target
```

- `--n-label-cols` tells the pipeline how many trailing columns are labels
  (default: 6, matching the original dataset).
- `--target-col` (or `--label-col` for `run_pipeline.py`) tells it which of
  those label columns is the actual regression target. If omitted, the
  **last** label column is used (matching the original notebook's
  `train.iloc[:, -1]` behaviour).
- Train and test files must have the same feature columns; if they don't,
  the pipeline restricts to the common columns and warns you.

If the other (non-target) label columns are alternate targets rather than
real features — as in the original dataset — they are automatically
excluded from the feature matrix at training time (see
[Bug fixes](#bug-fixes-from-the-original-notebook), bug #3).

## Usage

### 1. Feature selection only

```bash
python feature_selection_pipeline.py \
    --train  data/train_lab.COM \
    --test   data/test_lab.COM \
    --n-label-cols 6 \
    --feature-set-sizes 100 200 400 600 800 \
    --methods lasso elasticnet randomforest lgbm informationgain \
              relieff rfe_svm rfe_rf mrmr kbest_f boruta \
    --output-dir Feature_selection \
    --feature-sets-json feature_sets.json
```

This writes, for every method `<m>` and size `<k>`:

```
Feature_selection/<m>/<m>_top_<k>_train.csv
Feature_selection/<m>/<m>_top_<k>_test.csv
Feature_selection/<m>/<m>_feature_sets.csv     # long-format list of selected features
feature_sets.json                              # combined summary across all methods/sizes
```

Run `python feature_selection_pipeline.py --help` for the full list of
options, including per-method hyperparameters (`--rf-n-estimators`,
`--rfe-step`, `--boruta-max-iter`, `--cv-folds`, `--random-state`, etc.).

### 2. Model training only (on one feature set)

```bash
python auto_new1_scaler.py \
    -tr Feature_selection/lasso/lasso_top_200_train.csv \
    -v  Feature_selection/lasso/lasso_top_200_test.csv \
    -l  hl_10 \
    -dc lab_a lab_b lab_c lab_d lab_e \
    -m  RFR \
    -p  Y \
    -s  Y \
    -f  top_200 \
    -pf params.csv \
    -o  results.csv \
    -od trainer_output/
```

| Flag | Meaning |
|---|---|
| `-tr / --train` | Training CSV. |
| `-v / --valid` | Validation/test CSV. |
| `-l / --lab` | Name of the target column. |
| `-dc / --drop-cols` | Other columns to exclude from the feature matrix (e.g. alternate label columns). Space-separated. |
| `-m / --ml` | Algorithm: `DTR`, `RFR`, `LR`, `LAS`, `RID`, `ENR`, `SVR`, `MLP`, `AD`, `GBR`, `XGB`, or `ALL`. |
| `-p / --popt` | `Y`/`N` — run `GridSearchCV` hyperparameter tuning. |
| `-s / --scale` | `Y`/`N` — `StandardScaler`-normalize features. |
| `-k / --kfold` | Number of CV folds (default 5). |
| `-f / --feature` | Label used in output filenames (e.g. `top_200`). |
| `-pf / --pfile` | Output path for the chosen hyperparameters. |
| `-o / --outputfile` | Output path for performance metrics. |
| `-od / --outdir` | Directory for per-model prediction CSVs. |

Outputs `<algo>_<feature>_train_model.csv` (5-fold CV out-of-fold
predictions) and `<algo>_<feature>_test_model.csv` (held-out validation
predictions), plus the parameter and metrics files.

### 3. Parallel training across all algorithms (one feature set)

```bash
bash Auto_bash.sh <RES_DIR> <FOLDER_PREFIX> <TRAIN_CSV> <VAL_CSV> <LABEL> <PARAM_OPT[Y/N]> \
                   [SCALE[Y/N]] [LIST_FILE] [KFOLD] [DROP_COLS]
```

Example:

```bash
bash Auto_bash.sh result_top200 folder_top200_ \
    Feature_selection/lasso/lasso_top_200_train.csv \
    Feature_selection/lasso/lasso_top_200_test.csv \
    hl_10 N Y list 5 "lab_a lab_b lab_c lab_d lab_e"
```

`Auto_bash.sh` reads the `list` file (one algorithm code per line, e.g.
`DTR` / `RFR` / `XGB`), launches `auto_new1_scaler.py` once per algorithm
**in parallel** as background jobs, and collects every algorithm's
`result.<ALGO>` and `param_<ALGO>.csv` into `<RES_DIR>/`.

### 4. Full end-to-end pipeline (feature selection + training, all methods × sizes)

```bash
python run_pipeline.py \
    --train data/train_lab.COM \
    --test  data/test_lab.COM \
    --n-label-cols 6 \
    --label-col hl_10 \
    --feature-set-sizes 100 200 400 600 800 \
    --methods lasso elasticnet randomforest kbest_f mrmr boruta \
    --param-opt N \
    --scale Y \
    --kfold 5 \
    --list-file list
```

This runs `feature_selection_pipeline.py` once, then loops over every
generated `<method>/<method>_<size>_train.csv` / `_test.csv` pair and calls
`Auto_bash.sh` on each (which fans out to every algorithm in `list` in
parallel). Use `--skip-training` to only run the feature-selection stage.

Run `python run_pipeline.py --help` for the complete option list — it
forwards the same FS hyperparameters as `feature_selection_pipeline.py`,
plus `--param-opt`, `--scale`, `--kfold`, `--auto-bash-script`, and
`--list-file` for the training stage.

## Output structure (full pipeline)

```
Feature_selection/
├── lasso/
│   ├── lasso_top_100_train.csv
│   ├── lasso_top_100_test.csv
│   ├── lasso_top_100_result/             # created by Auto_bash.sh
│   │   ├── result.RFR
│   │   ├── result.LAS
│   │   ├── param_RFR.csv
│   │   └── param_LAS.csv
│   ├── lasso_top_100_folder_RFR/         # per-algorithm working dir
│   │   ├── RFR_RFR_train_model.csv       # 5-fold CV predictions
│   │   └── RFR_RFR_test_model.csv        # validation predictions
│   ├── lasso_top_200_train.csv
│   ├── ...
│   └── lasso_feature_sets.csv
├── elasticnet/
│   └── ...
└── ...
feature_sets.json
```

Each `result.<ALGO>` CSV has columns:
`Name, MAE_tr, RMSE_tr, PCC_tr, R2_tr, MAE_te, RMSE_te, PCC_te, R2_te`
(`_tr` = 5-fold CV on the training set, `_te` = held-out validation set).

## Feature selection methods

| Key | Method | Notes |
|---|---|---|
| `lasso` | LassoCV | Ranked by absolute coefficient magnitude. |
| `elasticnet` | ElasticNetCV | Ranked by absolute coefficient magnitude. |
| `randomforest` | RandomForestRegressor | Ranked by impurity-based importance. |
| `lgbm` | LGBMRegressor | Ranked by gain-based importance. *(optional dependency)* |
| `informationgain` | `mutual_info_regression` | |
| `relieff` | ReliefF (skrebate) | *(optional dependency, slower)* |
| `rfe_svm` | RFE with linear SVR | One fit per requested size. |
| `rfe_rf` | RFE with RandomForestRegressor | One fit per requested size; uses a smaller forest for speed. |
| `mrmr` | Minimum Redundancy Maximum Relevance | *(optional dependency)* |
| `kbest_f` | `SelectKBest(f_regression)` | Runs on raw (unscaled) features — `f_regression` is scale-invariant. |
| `boruta` | BorutaPy | All-relevant selection on top of a RandomForest; ranking used to pick top-K. *(optional dependency)* |

All methods except `kbest_f` operate on `StandardScaler`-normalized
features by default (`--skip-scaling` disables this).

## Regression algorithms

`DTR` (Decision Tree), `RFR` (Random Forest), `LR` (Linear Regression),
`LAS` (Lasso), `RID` (Ridge), `ENR` (Elastic Net), `SVR`, `MLP`, `AD`
(AdaBoost), `GBR` (Gradient Boosting), `XGB` (XGBoost, optional). Use `-m ALL`
to run every available algorithm.

## Bug fixes from the original notebook

This pipeline started life as an exploratory Jupyter notebook. While
converting it, the following real correctness bugs were found and fixed:

1. **Every FS method except Lasso used Lasso's ranking.** In the original
   notebook, the loops for ElasticNet, RandomForest, LGBM, Information Gain,
   ReliefF, MRMR, KBest, and Boruta all mistakenly selected
   `sorted_coef_lasso.head(size)` (Lasso's top features) instead of their
   own computed ranking (`sorted_coef_en`, `sorted_importances_rf`, etc.).
   In effect, 8 of the 10 "different" feature sets being compared were
   actually identical to Lasso's. **Fixed:** each method now correctly uses
   its own ranking.

2. **Tuned hyperparameters were discarded.** In `auto_new1_scaler.py`, when
   `-p Y` (GridSearchCV) was requested, `clf.best_estimator_` was computed
   but never reused — the K-fold CV and the final model both re-fit a
   fresh, default-hyperparameter model (`models[algo]`) instead. Tuning had
   no effect on reported performance. **Fixed:** the tuned estimator's
   hyperparameters (via `sklearn.base.clone`) are now used for every
   subsequent fit.

3. **Alternate label columns leaking into the feature matrix.** The dataset
   stores several possible regression targets as trailing label columns;
   the FS pipeline correctly carries all of them through into its output
   CSVs (so you can pick any one as the target later), but the original
   training script only dropped the *one* target column it was told about,
   leaving the other label columns sitting in `X` as if they were real
   features (including non-numeric ones, which crashed numeric models
   outright). **Fixed:** added `-dc / --drop-cols` to `auto_new1_scaler.py`
   (and threaded it through `Auto_bash.sh` and `run_pipeline.py`) so all
   non-target label columns are explicitly excluded from the feature matrix.

4. **Path bugs in `Auto_bash.sh`.** The error message referenced a
   non-existent `listDT` file instead of `list`, and train/validation file
   paths were always assumed relative to the run directory, breaking when
   an absolute path was passed in (as `run_pipeline.py` does). **Fixed:**
   corrected the error message and made path resolution handle both
   absolute and relative inputs.

## Tips

- Start with a small `--feature-set-sizes` (e.g. `5 10`) and a short
  `--methods` / `list` to sanity-check your data format before running the
  full panel — RFE and Boruta in particular are slow on large feature sets.
- `rfe_svm`, `rfe_rf`, and `boruta` scale poorly with both the number of
  samples and features; expect these to dominate total runtime.
- `Auto_bash.sh` launches one background process per line in `list` with no
  concurrency cap — on a shared machine, keep `list` to a manageable number
  of algorithms (or split it into batches) to avoid oversubscribing CPU cores,
  especially when also feeding it `ALL` via `-m`.
