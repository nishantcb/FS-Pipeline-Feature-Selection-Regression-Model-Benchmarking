#!/usr/bin/env python3
"""
run_pipeline.py

End-to-end orchestrator: runs feature_selection_pipeline.py to generate
per-method, per-size train/test CSVs, then runs Auto_bash.sh (which in turn
calls auto_new1_scaler.py) on every generated CSV pair to train and evaluate
regression models.

This mirrors the original notebook's "Cell 4 -> Cell 5" workflow but is
fully parameterized and safe to re-run.

Example
-------
    python run_pipeline.py \\
        --train ../comp_lab_train/train_lab.COM \\
        --test  ../comp_lab_test/test_lab.COM \\
        --n-label-cols 6 \\
        --label-col hl_10 \\
        --feature-set-sizes 100 200 400 600 800 \\
        --methods lasso elasticnet randomforest kbest_f \\
        --param-opt N \\
        --scale N

Notes
-----
- Requires Auto_bash.sh, auto_new1_scaler.py, and a `list` file (one
  algorithm code per line, e.g. DTR/RFR/XGB) to be in the same directory as
  this script, or pass --auto-bash-script / --list-file explicitly.
- Set --skip-training to only run feature selection (equivalent to the
  original notebook's Cell 4 alone).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end feature selection + model training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = p.add_argument_group("Input / output")
    io_group.add_argument("--train", required=True, help="Path to the training CSV.")
    io_group.add_argument("--test", required=True, help="Path to the test/validation CSV.")
    io_group.add_argument("--n-label-cols", type=int, default=6)
    io_group.add_argument("--target-col", default=None, help="Target column for feature selection (default: last label column).")
    io_group.add_argument(
        "--label-col",
        required=False,
        default=None,
        help="Name of the label column used as the regression target during MODEL TRAINING "
        "(passed to auto_new1_scaler.py as -l). Defaults to --target-col, or the last label "
        "column if neither is given.",
    )
    io_group.add_argument("--output-dir", default="Feature_selection")
    io_group.add_argument("--feature-sets-json", default="feature_sets.json")

    fs_group = p.add_argument_group("Feature selection")
    fs_group.add_argument("--feature-set-sizes", type=int, nargs="+", default=[100, 200, 400, 600, 800])
    fs_group.add_argument(
        "--methods",
        nargs="+",
        default=["lasso", "elasticnet", "randomforest", "lgbm", "informationgain", "relieff",
                  "rfe_svm", "rfe_rf", "mrmr", "kbest_f", "boruta"],
    )
    fs_group.add_argument("--skip-scaling", action="store_true")

    train_group = p.add_argument_group("Model training")
    train_group.add_argument("--skip-training", action="store_true", help="Only run feature selection, skip model training.")
    train_group.add_argument("--param-opt", choices=["Y", "N"], default="N", help="Run GridSearchCV tuning (-p flag).")
    train_group.add_argument("--scale", choices=["Y", "N"], default="N", help="Scale features before model training (-s flag).")
    train_group.add_argument("--kfold", type=int, default=5)
    train_group.add_argument(
        "--auto-bash-script",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Auto_bash.sh"),
        help="Path to Auto_bash.sh.",
    )
    train_group.add_argument(
        "--list-file",
        default=None,
        help="Path to the algorithm list file. Defaults to 'list' next to Auto_bash.sh.",
    )

    hp_group = p.add_argument_group("FS method hyperparameters (forwarded to feature_selection_pipeline.py)")
    hp_group.add_argument("--random-state", type=int, default=42)
    hp_group.add_argument("--n-jobs", type=int, default=-1)
    hp_group.add_argument("--cv-folds", type=int, default=5)
    hp_group.add_argument("--max-iter", type=int, default=2000)
    hp_group.add_argument("--rf-n-estimators", type=int, default=100)
    hp_group.add_argument("--rfe-rf-n-estimators", type=int, default=50)
    hp_group.add_argument("--rfe-step", type=float, default=0.2)
    hp_group.add_argument("--relieff-n-neighbors", type=int, default=100)
    hp_group.add_argument("--boruta-n-estimators", type=int, default=500)
    hp_group.add_argument("--boruta-max-iter", type=int, default=100)

    return p


def run_feature_selection(args, script_dir: str) -> None:
    fs_script = os.path.join(script_dir, "feature_selection_pipeline.py")
    cmd = [
        sys.executable,
        fs_script,
        "--train", args.train,
        "--test", args.test,
        "--n-label-cols", str(args.n_label_cols),
        "--output-dir", args.output_dir,
        "--feature-sets-json", args.feature_sets_json,
        "--feature-set-sizes", *[str(s) for s in args.feature_set_sizes],
        "--methods", *args.methods,
        "--random-state", str(args.random_state),
        "--n-jobs", str(args.n_jobs),
        "--cv-folds", str(args.cv_folds),
        "--max-iter", str(args.max_iter),
        "--rf-n-estimators", str(args.rf_n_estimators),
        "--rfe-rf-n-estimators", str(args.rfe_rf_n_estimators),
        "--rfe-step", str(args.rfe_step),
        "--relieff-n-neighbors", str(args.relieff_n_neighbors),
        "--boruta-n-estimators", str(args.boruta_n_estimators),
        "--boruta-max-iter", str(args.boruta_max_iter),
    ]
    if args.target_col:
        cmd += ["--target-col", args.target_col]
    if args.skip_scaling:
        cmd += ["--skip-scaling"]

    print("\n>>> Step 1/2: Feature selection")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def run_model_training(args, label_col: str, other_label_cols: list[str]) -> None:
    if not os.path.isfile(args.auto_bash_script):
        raise FileNotFoundError(
            f"Auto_bash.sh not found at {args.auto_bash_script}. Pass --auto-bash-script to point at it, "
            "or use --skip-training to run feature selection only."
        )

    print("\n>>> Step 2/2: Model training (per method, per feature-set size)")

    for method in args.methods:
        method_dir = os.path.join(args.output_dir, method)
        train_files = sorted(glob.glob(os.path.join(method_dir, f"{method}*_train.csv")))
        if not train_files:
            print(f"  [{method}] no train CSVs found in {method_dir}, skipping.")
            continue

        for train_file in train_files:
            test_file = train_file.replace("_train.csv", "_test.csv")
            if not os.path.isfile(test_file):
                print(f"  Skipping (no matching test file): {train_file}")
                continue

            name = os.path.basename(train_file).replace("_train.csv", "")
            res_dir_name = f"{name}_result"
            folder_prefix = f"{name}_folder_"

            cmd = [
                "bash",
                args.auto_bash_script,
                res_dir_name,
                folder_prefix,
                os.path.abspath(train_file),
                os.path.abspath(test_file),
                label_col,
                args.param_opt,
                args.scale,
                os.path.abspath(args.list_file) if args.list_file else "",
                str(args.kfold),
                " ".join(other_label_cols),
            ]

            print(f"\n  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=method_dir)
            if result.returncode != 0:
                print(f"  Warning: training run for {name} exited with code {result.returncode}")


def compute_other_label_cols(train_path: str, n_label_cols: int, label_col: str) -> list[str]:
    """Read just the header of the original train file to find which trailing
    label columns are NOT the chosen target — these ride along in every FS
    output CSV and must be excluded from the feature matrix at training time."""
    import pandas as pd

    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    label_block = header[-n_label_cols:]
    return [c for c in label_block if c != label_col]


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_feature_selection(args, script_dir)

    if args.skip_training:
        print("\n--skip-training set: stopping after feature selection.")
        return

    label_col = args.label_col or args.target_col
    if not label_col:
        raise SystemExit(
            "Model training requires a label column name. Pass --label-col explicitly "
            "(it cannot be inferred automatically once features are written out as plain CSVs)."
        )

    other_label_cols = compute_other_label_cols(args.train, args.n_label_cols, label_col)
    if other_label_cols:
        print(f"\nNote: the following alternate label columns will be excluded from the "
              f"feature matrix during training: {other_label_cols}")

    run_model_training(args, label_col, other_label_cols)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
