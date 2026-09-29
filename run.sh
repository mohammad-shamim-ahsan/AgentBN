#!/bin/bash

BENCHMARK=${1:-alarm}
SUBSET=${2:-1}

# Create experiment-specific log directory
mkdir -p "logs/$BENCHMARK"

# Run the pipeline and save the log
python3 orchestration_pipeline.py \
    --benchmark "$BENCHMARK" \
    --subset "$SUBSET" \
    > "logs/$BENCHMARK/subset_${SUBSET}_$(date +%Y%m%d_%H%M%S).log"