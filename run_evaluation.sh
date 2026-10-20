#!/bin/bash

set -o pipefail

BENCHMARK="${1:?Usage: $0 <alarm|der|lung_cancer>}"

case "$BENCHMARK" in
    alarm|der|lung_cancer)
        ;;
    *)
        echo "Error: Invalid benchmark '$BENCHMARK'"
        echo "Usage: $0 <alarm|der|lung_cancer>"
        exit 1
        ;;
esac

SCRIPT="datasets/automatic_reasoning_simple.py"
LOG_DIR="logs/$BENCHMARK"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evaluation_$TIMESTAMP.log"

echo "Benchmark: $BENCHMARK"
echo "Log file: $LOG_FILE"

python3 -u "$SCRIPT" > "$LOG_FILE" 2>&1

echo "Exit status: $?"
