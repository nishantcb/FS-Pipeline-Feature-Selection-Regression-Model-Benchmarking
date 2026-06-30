#!/usr/bin/env python3
"""
auto_new1_scaler.py

Trains and evaluates a regression model (one of several supported algorithms)
on a feature-selected train/validation CSV pair, with optional StandardScaler
normalization and optional GridSearchCV hyperparameter tuning.

This script is normally invoked once per algorithm by Auto_bash.sh (which
parallelizes across the algorithms listed in `list`), but it can equally be
run standalone for a single algorithm.

Outputs (written to --outdir, default: current directory)
-----------------------------------------------------------
  <algo>_<feature>_train_model.csv   5-fold CV out-of-fold predictions on train
  <algo>_<feature>_test_model.csv    Predictions on the validation/test file
  <pfile>                            Best hyperparameters found (if -p Y)
  <outputfile>                       MAE/RMSE/PCC/R2 for train (CV) and test

Example
-------
    python auto_new1_scaler.py \\
        -tr train_features.csv -v test_features.csv \\
        -l target -m RFR -p Y -k 5 -s Y \\
        -f top_200 -pf param_RFR.csv -o result.RFR
"""

from __future__ import print_function, division

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor as RFR,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DTR

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ===================== Metrics ===================== #
def perf_measure(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, pcc, r2


def build_model_registry():
    """Fresh, unfitted instances of every supported algorithm."""
    models = {
        "DTR": DTR(random_state=42),
        "RFR": RFR(random_state=42, n_jobs=-1),
        "SVR": SVR(),
        "MLP": MLPRegressor(random_state=42),
        "AD": AdaBoostRegressor(random_state=42),
        "GBR": GradientBoostingRegressor(random_state=42),
        "LAS": Lasso(random_state=42),
        "RID": Ridge(random_state=42),
        "LR": LinearRegression(n_jobs=-1),
        "ENR": ElasticNet(random_state=42, max_iter=20000),
    }
    if XGB_AVAILABLE:
        models["XGB"] = XGBRegressor(
            n_jobs=-1,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
        )
    return models


def build_param_grids():
    return {
        "DTR": {
            "criterion": ["squared_error", "absolute_error"],
            "min_samples_split": [10, 20, 40],
            "max_depth": [2, 6, 8],
            "min_samples_leaf": [20, 40, 100],
            "max_leaf_nodes": [5, 20, 100],
        },
        "RFR": {
            "n_estimators": [50, 100],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True],
        },
        "SVR": {
            "kernel": ["rbf", "linear"],
            "C": [1, 5, 10],
            "gamma": ["scale", "auto"],
            "epsilon": [0.1, 0.2, 0.3],
        },
        "MLP": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "max_iter": [5000],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
        },
        "AD": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
        "GBR": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]},
        "LAS": {"alpha": [0.01, 0.1, 1, 10], "max_iter": [5000, 7000, 9000]},
        "RID": {"alpha": [0.01, 0.1, 1, 10]},
        "ENR": {
            "alpha": [0.001, 0.01, 0.1, 1, 10],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            "max_iter": [5000, 20000],
        },
    }


# ===================== ML Core Process ===================== #
def ML(train_file, valid_file, label, p_opt, kfolds, algo, feat, scale_opt, outdir, drop_cols=None):
    models = build_model_registry()
    param_grid = build_param_grids()

    if algo not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(f"Unknown/unavailable algorithm '{algo}'. Available: {available}")

    # Load data
    train = pd.read_csv(train_file)
    valid = pd.read_csv(valid_file)

    if label not in train.columns:
        raise KeyError(f"Label column '{label}' not found in training file ({train_file}).")
    if label not in valid.columns:
        raise KeyError(f"Label column '{label}' not found in validation file ({valid_file}).")

    # Columns to exclude from the feature matrix entirely (e.g. alternate
    # target/label columns that ride along in the file but aren't the
    # current regression target and aren't real features). The target
    # column itself is always excluded regardless of drop_cols.
    cols_to_drop = set(drop_cols or [])
    cols_to_drop.add(label)
    cols_to_drop_train = [c for c in cols_to_drop if c in train.columns]
    cols_to_drop_valid = [c for c in cols_to_drop if c in valid.columns]

    X = train.drop(columns=cols_to_drop_train)
    y = train[label]

    X_val = valid.drop(columns=cols_to_drop_valid)
    y_val = valid[label]

    if list(X.columns) != list(X_val.columns):
        common = [c for c in X.columns if c in set(X_val.columns)]
        warnings.warn(
            "Train and validation feature columns differ after dropping label columns. "
            f"Restricting to the {len(common)} common columns."
        )
        X = X[common]
        X_val = X_val[common]

    # ========= OPTIONAL SCALING ========= #
    if scale_opt == "Y":
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    base_model = models[algo]

    # ===================== Parameter Optimization ===================== #
    # If tuning is requested and a grid exists for this algo, GridSearchCV
    # finds the best hyperparameters. Those hyperparameters (NOT just the
    # single fitted estimator) are then reused as the template for every
    # subsequent fit (K-fold CV folds + final full-train model), so the
    # tuning actually has an effect downstream.
    if p_opt == "Y" and algo in param_grid:
        clf = GridSearchCV(base_model, param_grid[algo], cv=kfolds, n_jobs=-1)
        clf.fit(X, y)
        best_model = clf.best_estimator_
        model_template = clf.best_estimator_
    else:
        best_model = base_model
        model_template = base_model

    # ===================== K-Fold (out-of-fold predictions) ===================== #
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    preds_all = []
    true_all = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model_template)
        fold_model.fit(X_tr, y_tr)

        preds_all.extend(fold_model.predict(X_te))
        true_all.extend(y_te)

    cv_df = pd.DataFrame({"true": true_all, "pred": preds_all})
    cv_path = os.path.join(outdir, f"{algo}_{feat}_train_model.csv")
    cv_df.to_csv(cv_path, index=False)

    # ===================== Retrain Final Model on Full Train ===================== #
    final_model = clone(model_template)
    final_model.fit(X, y)

    # ===================== Validation ===================== #
    val_preds = final_model.predict(X_val)

    val_df = pd.DataFrame(
        {
            "ID": valid.index,
            "pred": val_preds,
            "true": y_val.values,
        }
    )
    val_path = os.path.join(outdir, f"{algo}_{feat}_test_model.csv")
    val_df.to_csv(val_path, index=False)

    return algo, feat, str(best_model), cv_path, val_path


# ===================== MAIN ===================== #
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a regression model on a feature-selected dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-tr", "--train", required=True, help="Path to training CSV.")
    parser.add_argument("-v", "--valid", required=True, help="Path to validation/test CSV.")
    parser.add_argument("-l", "--lab", required=True, help="Name of the target/label column.")
    parser.add_argument("-k", "--kfold", type=int, default=5, help="Number of K-fold CV splits.")
    parser.add_argument(
        "-p",
        "--popt",
        type=str.upper,
        choices=["Y", "N"],
        default="N",
        help="Run GridSearchCV hyperparameter tuning (Y/N).",
    )
    parser.add_argument(
        "-m",
        "--ml",
        type=str.upper,
        choices=["DTR", "RFR", "LR", "LAS", "RID", "ENR", "SVR", "MLP", "AD", "GBR", "XGB", "ALL"],
        default="DTR",
        help="Algorithm to run, or ALL to run every supported algorithm.",
    )
    parser.add_argument("-f", "--feature", default="New", help="Label used in output filenames (e.g. top_200).")
    parser.add_argument("-pf", "--pfile", default="Parameters_file.csv", help="Output path for best hyperparameters.")
    parser.add_argument("-o", "--outputfile", default="Best_results.csv", help="Output path for performance metrics.")
    parser.add_argument(
        "-s",
        "-S",
        "--scale",
        type=str.upper,
        choices=["Y", "N"],
        default="N",
        help="Normalize features using StandardScaler (Y/N).",
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default=".",
        help="Directory to write per-model prediction CSVs into.",
    )
    parser.add_argument(
        "-dc",
        "--drop-cols",
        nargs="*",
        default=None,
        help="Extra columns to exclude from the feature matrix in addition to the target "
        "label (e.g. other alternate-target/label columns that should never be treated "
        "as features). Space-separated list of column names.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    print("\n================ Regression Pipeline Summary ================")
    print("Train:", args.train)
    print("Valid:", args.valid)
    print("Label:", args.lab)
    print("ML:", args.ml)
    print("Feature:", args.feature)
    print("K-Fold:", args.kfold)
    print("Parameter Opt:", args.popt)
    print("Scaling:", args.scale)
    print("Out dir:", args.outdir)
    print("============================================================\n")

    if args.ml == "ALL":
        algos = ["DTR", "RFR", "LR", "LAS", "RID", "ENR", "SVR", "MLP", "AD", "GBR"]
        if XGB_AVAILABLE:
            algos.append("XGB")
        else:
            print("Warning: 'xgboost' not installed. Skipping XGB in ALL mode.\n")
    else:
        if args.ml == "XGB" and not XGB_AVAILABLE:
            raise SystemExit("Error: -m XGB requested but the 'xgboost' package is not installed.")
        algos = [args.ml]

    results = []
    params = []

    for a in algos:
        print(f"Running {a} ...")
        method, feat, param, cv_path, val_path = ML(
            args.train, args.valid, args.lab, args.popt, args.kfold, a, args.feature, args.scale,
            args.outdir, drop_cols=args.drop_cols
        )

        params.append([method, feat, param])

        train_df = pd.read_csv(cv_path)
        test_df = pd.read_csv(val_path)

        tr = perf_measure(train_df["true"], train_df["pred"])
        te = perf_measure(test_df["true"], test_df["pred"])

        res = list(tr) + list(te)
        results.append([a + "_" + args.feature] + res)
        print(f"  -> MAE_te={te[0]:.4f}  RMSE_te={te[1]:.4f}  PCC_te={te[2]:.4f}  R2_te={te[3]:.4f}")

    # save parameter file
    pd.DataFrame(params, columns=["Method", "Features", "Parameters"]).to_csv(args.pfile, index=False)

    # save results
    cols = ["Name", "MAE_tr", "RMSE_tr", "PCC_tr", "R2_tr", "MAE_te", "RMSE_te", "PCC_te", "R2_te"]
    pd.DataFrame(results, columns=cols).to_csv(args.outputfile, index=False)

    print(f"\nParameters saved to: {args.pfile}")
    print(f"Results saved to:    {args.outputfile}")
    print("DONE.")


if __name__ == "__main__":
    main()
