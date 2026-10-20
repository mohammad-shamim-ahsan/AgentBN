#!/bin/bash

set -o pipefail

BENCHMARK="${1:?Usage: $0 <alarm|der|lung_cancer> <reasoning|validation>}"
MODE="${2:?Usage: $0 <alarm|der|lung_cancer> <reasoning|validation>}"

case "$BENCHMARK" in
    alarm|der|lung_cancer)
        ;;
    *)
        echo "Error: Invalid benchmark '$BENCHMARK'"
        exit 1
        ;;
esac

case "$MODE" in
    reasoning)
        SCRIPT="datasets/automatic_reasoning_simple.py"
        ;;
    validation)
        SCRIPT="evaluation/bn_validator.py"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Usage: $0 <alarm|der|lung_cancer> <reasoning|validation>"
        exit 1
        ;;
esac

LOG_DIR="logs/$BENCHMARK"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/${MODE}_$TIMESTAMP.log"

echo "Benchmark: $BENCHMARK"
echo "Mode: $MODE"
echo "Script: $SCRIPT"
echo "Log file: $LOG_FILE"

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python3 -u "$SCRIPT" > "$LOG_FILE" 2>&1

echo "Exit status: $?"
