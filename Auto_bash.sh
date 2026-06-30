#!/bin/bash
#
# Auto_bash.sh
#
# Parallel orchestrator for auto_new1_scaler.py. Reads a newline-separated
# list of algorithm codes from a "list" file (one of DTR, RFR, LR, LAS, RID,
# ENR, SVR, MLP, AD, GBR, XGB, ALL per line) and launches one
# auto_new1_scaler.py run per algorithm in the background, in parallel.
#
# Usage:
#   bash Auto_bash.sh <RES_DIR_NAME> <FOLDER_PREFIX> <TRAIN_FILE> <VAL_FILE> <LABEL> <PARAM_OPT[Y/N]> [SCALE[Y/N]] [LIST_FILE] [KFOLD] [DROP_COLS]
#
# Required positional arguments:
#   RES_DIR_NAME    Name of the results directory to create under the run dir.
#   FOLDER_PREFIX   Prefix used for the per-algorithm working folders.
#   TRAIN_FILE      Path (relative to the run dir) to the training CSV.
#   VAL_FILE        Path (relative to the run dir) to the validation/test CSV.
#   LABEL           Name of the target/label column in TRAIN_FILE/VAL_FILE.
#   PARAM_OPT       Y or N — whether to run GridSearchCV hyperparameter tuning.
#
# Optional positional arguments:
#   SCALE           Y or N — whether to StandardScaler-normalize features (default: N).
#   LIST_FILE       Path to the algorithm list file (default: <script_dir>/list).
#   KFOLD           Number of K-fold CV splits (default: 5).
#   DROP_COLS       Space-separated, quoted string of extra column names to exclude from
#                    the feature matrix (e.g. alternate label columns). Default: none.
#
# Example:
#   bash Auto_bash.sh result_top200 folder_top200_ train_top200.csv test_top200.csv hl_10 N Y "" 5 "lab_a lab_b lab_c lab_d lab_e"
#
set -uo pipefail

# -------------------------------
# Validate input arguments
# -------------------------------
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <RES_DIR_NAME> <FOL_PREFIX> <TRAIN_FILE> <VAL_FILE> <LABEL> <PARAM_OPT[Y/N]> [SCALE[Y/N]] [LIST_FILE] [KFOLD]"
  exit 1
fi

RES="$1"
FOL="$2"
TRAIN_FILE="$3"
VAL_FILE="$4"
LABEL="$5"
PARA="$6"
SCALE="${7:-N}"
LIST_FILE_OVERRIDE="${8:-}"
KFOLD="${9:-5}"
DROP_COLS="${10:-}"

# 🔥 SCRIPT DIRECTORY (where this .sh file exists — auto_new1_scaler.py is
# expected to live alongside it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 🔥 RUN DIRECTORY (where you run the script from)
RUN_DIR="$(pwd)"

LIST_FILE="${LIST_FILE_OVERRIDE:-$SCRIPT_DIR/list}"

# TRAIN_FILE / VAL_FILE may be given as absolute paths or as paths relative
# to RUN_DIR — resolve to an absolute path either way.
if [[ "$TRAIN_FILE" = /* ]]; then
  TRAIN_FILE_ABS="$TRAIN_FILE"
else
  TRAIN_FILE_ABS="$RUN_DIR/$TRAIN_FILE"
fi
if [[ "$VAL_FILE" = /* ]]; then
  VAL_FILE_ABS="$VAL_FILE"
else
  VAL_FILE_ABS="$RUN_DIR/$VAL_FILE"
fi

# -------------------------------
# Create results directory in RUN DIR
# -------------------------------
RESULTS_DIR="$RUN_DIR/$RES"
mkdir -p "$RESULTS_DIR"

echo "📌 Script directory: $SCRIPT_DIR"
echo "📌 Run directory:    $RUN_DIR"
echo "📋 Algorithm list:   $LIST_FILE"
echo "📂 Saving all results inside: $RESULTS_DIR"
echo ""

# -------------------------------
# Check essential files exist
# -------------------------------
if [[ ! -f "$SCRIPT_DIR/auto_new1_scaler.py" ]]; then
  echo "❌ Error: auto_new1_scaler.py not found at $SCRIPT_DIR/auto_new1_scaler.py"
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "❌ Error: algorithm list file not found at $LIST_FILE"
  echo "   Create a plain text file with one algorithm code per line, e.g.:"
  echo "     DTR"
  echo "     RFR"
  echo "     XGB"
  exit 1
fi

if [[ ! -f "$TRAIN_FILE_ABS" ]]; then
  echo "❌ Error: Training file '$TRAIN_FILE_ABS' not found!"
  exit 1
fi

if [[ ! -f "$VAL_FILE_ABS" ]]; then
  echo "❌ Error: Validation file '$VAL_FILE_ABS' not found!"
  exit 1
fi

# -------------------------------
# Main loop — one background job per algorithm
# -------------------------------
while read -r line
do
  # Skip blank lines and comments
  [[ -z "$line" || "$line" =~ ^# ]] && continue

(
    folder_path="$RUN_DIR/${FOL}${line}"
    mkdir -p "$folder_path"

    echo -e "\n📂 Processing: $line"
    echo "📁 Folder: $folder_path"

    time python3 "$SCRIPT_DIR/auto_new1_scaler.py" \
        -tr "$TRAIN_FILE_ABS" \
        -v "$VAL_FILE_ABS" \
        -k "$KFOLD" \
        -m "$line" \
        -l "$LABEL" \
        -p "$PARA" \
        -s "$SCALE" \
        -pf "$RESULTS_DIR/param_$line.csv" \
        -f "$line" \
        -o "$RESULTS_DIR/result.$line" \
        -od "$folder_path" \
        ${DROP_COLS:+-dc $DROP_COLS}

    status=$?
    if [[ $status -eq 0 ]]; then
      echo "✅ Completed: $line"
    else
      echo "❌ Failed: $line (exit code $status)"
    fi

) &
done < "$LIST_FILE"

wait

echo -e "\n🎉 All tasks finished! Results saved in: $RESULTS_DIR"
