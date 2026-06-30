#!/usr/bin/env python3
"""
feature_selection_pipeline.py

Runs a panel of feature selection (FS) methods on a regression dataset and
saves, for every method and every requested feature-set size, a train/test
CSV containing only the selected features plus the original label columns.

Methods included
-----------------
  lasso             LassoCV (coefficient magnitude)
  elasticnet        ElasticNetCV (coefficient magnitude)
  randomforest      RandomForestRegressor (impurity importance)
  lgbm              LGBMRegressor (gain importance)               [optional]
  informationgain   mutual_info_regression
  relieff           skrebate ReliefF                              [optional]
  rfe_svm           Recursive Feature Elimination, SVR(linear) estimator
  rfe_rf            Recursive Feature Elimination, RandomForest estimator
  mrmr              Minimum Redundancy Maximum Relevance           [optional]
  kbest_f           SelectKBest with f_regression
  boruta            BorutaPy (all-relevant feature selection)      [optional]

Methods marked [optional] are skipped automatically (with a warning) if the
corresponding package is not installed.

Input data format
------------------
Both the train and test/validation CSVs are expected to follow the same
layout used in the original project: a block of feature columns followed by
a block of label columns at the end, e.g.

    feat_1, feat_2, ..., feat_N, lab_other1, lab_other2, ..., lab_target

By default the script assumes the LAST 6 columns are label columns and that
the very last column is the regression target used to drive feature
selection. Both numbers are configurable (--n-label-cols, and the target is
always the last of those label columns unless --target-col is given).

Example
-------
    python feature_selection_pipeline.py \\
        --train ../comp_lab_train/train_lab.COM \\
        --test  ../comp_lab_test/test_lab.COM \\
        --n-label-cols 6 \\
        --feature-set-sizes 100 200 400 600 800 \\
        --methods lasso elasticnet randomforest kbest_f \\
        --output-dir Feature_selection \\
        --n-jobs -1

Output
------
For every method `<m>` and size `<k>`:
    <output-dir>/<m>/<m>_top_<k>_train.csv
    <output-dir>/<m>/<m>_top_<k>_test.csv
    <output-dir>/<m>/<m>_feature_sets.csv       (long-format summary)

And a single combined JSON with every method's selected features at every
size:
    <output-dir>/<feature-sets-json>            (default: feature_sets.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ALL_METHODS = [
    "lasso",
    "elasticnet",
    "randomforest",
    "lgbm",
    "informationgain",
    "relieff",
    "rfe_svm",
    "rfe_rf",
    "mrmr",
    "kbest_f",
    "boruta",
]

# Methods that depend on an optional third-party package.
OPTIONAL_METHODS = {"lgbm", "relieff", "mrmr", "boruta"}


# --------------------------------------------------------------------------- #
# Optional dependency imports — every one is wrapped so the script degrades
# gracefully (skip + warn) instead of crashing if a package is missing.
# --------------------------------------------------------------------------- #
def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401

        return lgb
    except ImportError:
        return None


def _try_import_relieff():
    try:
        from skrebate import ReliefF

        return ReliefF
    except ImportError:
        return None


def _try_import_mrmr():
    try:
        from mrmr import mrmr_regression

        return mrmr_regression
    except ImportError:
        return None


def _try_import_boruta():
    try:
        from boruta import BorutaPy

        return BorutaPy
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_dataset(path: str, n_label_cols: int):
    """Split a CSV into features (X) and the trailing label block (y_block)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    if n_label_cols <= 0 or n_label_cols >= df.shape[1]:
        raise ValueError(
            f"--n-label-cols={n_label_cols} is invalid for a file with "
            f"{df.shape[1]} columns ({path}). It must be between 1 and "
            f"(n_columns - 1)."
        )
    X = df.iloc[:, :-n_label_cols].copy()
    y_block = df.iloc[:, -n_label_cols:].copy()
    return X, y_block


def resolve_target(y_block: pd.DataFrame, target_col: Optional[str]) -> pd.Series:
    """Pick the regression target used to drive feature selection."""
    if target_col is not None:
        if target_col not in y_block.columns:
            raise ValueError(
                f"--target-col '{target_col}' was not found among the label "
                f"columns: {list(y_block.columns)}"
            )
        return y_block[target_col]
    # Default: the last label column, matching the original notebook's
    # behaviour (train.iloc[:, -1]).
    return y_block.iloc[:, -1]


def save_feature_outputs(
    method_name: str,
    feature_dict: Dict[str, List[str]],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_block: pd.DataFrame,
    y_test_block: pd.DataFrame,
    base_dir: str,
) -> None:
    """Write per-size train/test CSVs and a summary CSV for one FS method."""
    method_dir = os.path.join(base_dir, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)

    rows = []
    for set_name, features in feature_dict.items():
        missing = [f for f in features if f not in X_train.columns]
        if missing:
            raise KeyError(
                f"[{method_name}] selected features not present in X: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        X_sel = X_train[features]
        X_te_sel = X_test[features]

        train_path = os.path.join(method_dir, f"{method_name.lower()}_{set_name}_train.csv")
        test_path = os.path.join(method_dir, f"{method_name.lower()}_{set_name}_test.csv")

        pd.concat([X_sel.reset_index(drop=True), y_train_block.reset_index(drop=True)], axis=1).to_csv(
            train_path, index=False
        )
        pd.concat([X_te_sel.reset_index(drop=True), y_test_block.reset_index(drop=True)], axis=1).to_csv(
            test_path, index=False
        )

        print(
            f"  [{method_name}] {set_name}: saved {train_path} ({X_sel.shape}) "
            f"| {test_path} ({X_te_sel.shape})"
        )
        for f in features:
            rows.append({"method": method_name, "set": set_name, "feature": f})

    summary_path = os.path.join(method_dir, f"{method_name.lower()}_feature_sets.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  [{method_name}] feature summary saved -> {summary_path}")


# --------------------------------------------------------------------------- #
# Individual FS methods.
# Each returns {"top_<size>": [feature_name, ...], ...} for every requested size.
# --------------------------------------------------------------------------- #
def fs_lasso(X_scaled, y, sizes, n_jobs, cv, random_state, max_iter):
    model = LassoCV(cv=cv, random_state=random_state, n_jobs=n_jobs, max_iter=max_iter).fit(X_scaled, y)
    ranked = pd.Series(model.coef_, index=X_scaled.columns).abs().sort_values(ascending=False)
    return {f"top_{k}": ranked.head(k).index.tolist() for k in sizes}


def fs_elasticnet(X_scaled, y, sizes, n_jobs, cv, random_state, max_iter):
    model = ElasticNetCV(cv=cv, random_state=random_state, n_jobs=n_jobs, max_iter=max_iter).fit(X_scaled, y)
    ranked = pd.Series(model.coef_, index=X_scaled.columns).abs().sort_values(ascending=False)
    return {f"top_{k}": ranked.head(k).index.tolist() for k in sizes}


def fs_randomforest(X_scaled, y, sizes, n_jobs, n_estimators, random_state):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_scaled, y)
    ranked = pd.Series(model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    return {f"top_{k}": ranked.head(k).index.tolist() for k in sizes}


def fs_lgbm(X_scaled, y, sizes, n_jobs, random_state):
    lgb = _try_import_lightgbm()
    if lgb is None:
        return None
    model = lgb.LGBMRegressor(random_state=random_state, n_jobs=n_jobs, verbosity=-1)
    model.fit(X_scaled, y)
    ranked = pd.Series(model.feature_importances_, index=X_scaled.columns).sort_values(ascending=False)
    return {f"top_{k}": ranked.head(k).index.tolist() for k in sizes}


def fs_informationgain(X_scaled, y, sizes, random_state):
    ig = mutual_info_regression(X_scaled, y, random_state=random_state)
    ranked = pd.Series(ig, index=X_scaled.columns).sort_values(ascending=False)
    return {f"top_{k}": ranked.head(k).index.tolist() for k in sizes}


def fs_relieff(X_scaled, y, sizes, n_jobs, n_neighbors):
    ReliefF = _try_import_relieff()
    if ReliefF is None:
        return None
    relief = ReliefF(n_features_to_select=max(sizes), n_neighbors=n_neighbors, n_jobs=n_jobs)
    relief.fit(X_scaled.values, y.values)
    ranked_cols = [X_scaled.columns[i] for i in relief.top_features_]
    return {f"top_{k}": ranked_cols[:k] for k in sizes}


def fs_rfe_svm(X_scaled, y, sizes, step):
    out = {}
    for k in sizes:
        svm = SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=k, step=step)
        rfe.fit(X_scaled, y)
        idx = rfe.get_support(indices=True)
        out[f"top_{k}"] = X_scaled.columns[idx].tolist()
    return out


def fs_rfe_rf(X_scaled, y, sizes, step, n_estimators, n_jobs, random_state):
    out = {}
    for k in sizes:
        est = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
        rfe = RFE(estimator=est, n_features_to_select=k, step=step)
        rfe.fit(X_scaled, y)
        idx = rfe.get_support(indices=True)
        out[f"top_{k}"] = X_scaled.columns[idx].tolist()
    return out


def fs_mrmr(X_scaled, y, sizes):
    mrmr_regression = _try_import_mrmr()
    if mrmr_regression is None:
        return None
    out = {}
    for k in sizes:
        selected = mrmr_regression(X=pd.DataFrame(X_scaled, columns=X_scaled.columns), y=y, K=k)
        out[f"top_{k}"] = selected
    return out


def fs_kbest_f(X, y, sizes):
    out = {}
    for k in sizes:
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        idx = selector.get_support(indices=True)
        out[f"top_{k}"] = X.columns[idx].tolist()
    return out


def fs_boruta(X_scaled, y, sizes, n_estimators, random_state, max_iter):
    BorutaPy = _try_import_boruta()
    if BorutaPy is None:
        return None
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    boruta = BorutaPy(
        estimator=rf, n_estimators="auto", verbose=0, random_state=random_state, max_iter=max_iter
    )
    boruta.fit(np.array(X_scaled), np.array(y))
    ranks = pd.Series(boruta.ranking_, index=X_scaled.columns).sort_values()
    ranked_cols = ranks.index.tolist()
    return {f"top_{k}": ranked_cols[:k] for k in sizes}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run multiple feature selection methods on a regression dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = p.add_argument_group("Input / output")
    io_group.add_argument("--train", required=True, help="Path to the training CSV.")
    io_group.add_argument("--test", required=True, help="Path to the test/validation CSV.")
    io_group.add_argument(
        "--n-label-cols",
        type=int,
        default=6,
        help="Number of trailing columns in --train/--test that are label columns "
        "(everything before this is treated as a feature).",
    )
    io_group.add_argument(
        "--target-col",
        default=None,
        help="Name of the label column to use as the regression target for feature "
        "selection. Defaults to the LAST label column (matches original behaviour).",
    )
    io_group.add_argument(
        "--output-dir",
        default="Feature_selection",
        help="Directory where per-method subfolders and CSVs are written.",
    )
    io_group.add_argument(
        "--feature-sets-json",
        default="feature_sets.json",
        help="Filename (relative to current directory) for the combined JSON summary.",
    )

    fs_group = p.add_argument_group("Feature selection")
    fs_group.add_argument(
        "--feature-set-sizes",
        type=int,
        nargs="+",
        default=[100, 200, 400, 600, 800],
        help="List of top-K feature counts to extract for each method.",
    )
    fs_group.add_argument(
        "--methods",
        nargs="+",
        default=ALL_METHODS,
        choices=ALL_METHODS,
        help="Which FS methods to run.",
    )
    fs_group.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip StandardScaler normalization before FS (kbest_f always uses "
        "raw features regardless of this flag, matching original behaviour).",
    )

    hp_group = p.add_argument_group("Method hyperparameters")
    hp_group.add_argument("--random-state", type=int, default=42)
    hp_group.add_argument("--n-jobs", type=int, default=-1, help="Parallelism for sklearn estimators.")
    hp_group.add_argument("--cv-folds", type=int, default=5, help="CV folds for LassoCV / ElasticNetCV.")
    hp_group.add_argument("--max-iter", type=int, default=2000, help="Max iterations for Lasso/ElasticNet.")
    hp_group.add_argument(
        "--rf-n-estimators", type=int, default=100, help="Number of trees for the RandomForest FS method."
    )
    hp_group.add_argument(
        "--rfe-rf-n-estimators",
        type=int,
        default=50,
        help="Number of trees per RFE-RF iteration (kept small for speed).",
    )
    hp_group.add_argument("--rfe-step", type=float, default=0.2, help="Step size (fraction) for RFE.")
    hp_group.add_argument("--relieff-n-neighbors", type=int, default=100)
    hp_group.add_argument(
        "--boruta-n-estimators", type=int, default=500, help="Trees for the base RF used inside Boruta."
    )
    hp_group.add_argument("--boruta-max-iter", type=int, default=100)

    return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    print("=" * 70)
    print("Feature Selection Pipeline")
    print("=" * 70)
    print(f"Train file        : {args.train}")
    print(f"Test file          : {args.test}")
    print(f"Label columns      : last {args.n_label_cols}")
    print(f"Target column      : {args.target_col or '(last label column)'}")
    print(f"Methods            : {', '.join(args.methods)}")
    print(f"Feature set sizes  : {args.feature_set_sizes}")
    print(f"Output directory   : {args.output_dir}")
    print("=" * 70 + "\n")

    X, y_train_block = load_dataset(args.train, args.n_label_cols)
    X_te, y_test_block = load_dataset(args.test, args.n_label_cols)

    if list(X.columns) != list(X_te.columns):
        common = [c for c in X.columns if c in set(X_te.columns)]
        warnings.warn(
            "Train and test feature columns do not match exactly. "
            f"Restricting to the {len(common)} common columns."
        )
        X = X[common]
        X_te = X_te[common]

    y = resolve_target(y_train_block, args.target_col)

    max_size = max(args.feature_set_sizes)
    if max_size > X.shape[1]:
        raise ValueError(
            f"Requested feature-set size {max_size} exceeds the number of available "
            f"features ({X.shape[1]})."
        )

    if args.skip_scaling:
        X_scaled = X.copy()
    else:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    os.makedirs(args.output_dir, exist_ok=True)
    feature_sets: Dict[str, Dict[str, List[str]]] = {}
    sizes = args.feature_set_sizes
    n_methods = len(args.methods)

    for i, method in enumerate(args.methods, start=1):
        print(f"[{i}/{n_methods}] Running {method} ...")
        result = None

        if method == "lasso":
            result = fs_lasso(X_scaled, y, sizes, args.n_jobs, args.cv_folds, args.random_state, args.max_iter)
        elif method == "elasticnet":
            result = fs_elasticnet(
                X_scaled, y, sizes, args.n_jobs, args.cv_folds, args.random_state, args.max_iter
            )
        elif method == "randomforest":
            result = fs_randomforest(X_scaled, y, sizes, args.n_jobs, args.rf_n_estimators, args.random_state)
        elif method == "lgbm":
            result = fs_lgbm(X_scaled, y, sizes, args.n_jobs, args.random_state)
            if result is None:
                print("  Warning: 'lightgbm' not installed. Skipping LGBM.")
        elif method == "informationgain":
            result = fs_informationgain(X_scaled, y, sizes, args.random_state)
        elif method == "relieff":
            result = fs_relieff(X_scaled, y, sizes, args.n_jobs, args.relieff_n_neighbors)
            if result is None:
                print("  Warning: 'skrebate' not installed. Skipping ReliefF.")
        elif method == "rfe_svm":
            result = fs_rfe_svm(X_scaled, y, sizes, args.rfe_step)
        elif method == "rfe_rf":
            result = fs_rfe_rf(
                X_scaled, y, sizes, args.rfe_step, args.rfe_rf_n_estimators, args.n_jobs, args.random_state
            )
        elif method == "mrmr":
            result = fs_mrmr(X_scaled, y, sizes)
            if result is None:
                print("  Warning: 'mrmr_selection' not installed. Skipping MRMR.")
        elif method == "kbest_f":
            # Matches original notebook: SelectKBest(f_regression) is run on
            # the RAW (unscaled) features, since f_regression is scale-invariant.
            result = fs_kbest_f(X, y, sizes)
        elif method == "boruta":
            result = fs_boruta(X_scaled, y, sizes, args.boruta_n_estimators, args.random_state, args.boruta_max_iter)
            if result is None:
                print("  Warning: 'Boruta' not installed. Skipping Boruta.")
        else:
            print(f"  Unknown method '{method}', skipping.")

        if result is None:
            continue

        feature_sets[method] = result
        save_feature_outputs(
            method_name=method,
            feature_dict=result,
            X_train=X,
            X_test=X_te,
            y_train_block=y_train_block,
            y_test_block=y_test_block,
            base_dir=args.output_dir,
        )
        print()

    json_path = args.feature_sets_json
    with open(json_path, "w") as f:
        json.dump(feature_sets, f, indent=4)

    print("=" * 70)
    print(f"Feature selection complete. Ran {len(feature_sets)}/{n_methods} requested methods.")
    print(f"Combined feature sets saved to: {json_path}")
    print(f"Per-method CSVs saved under:    {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
