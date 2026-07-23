#!/bin/bash

BENCHMARK=${1:-alarm}

# Create experiment-specific log directory
mkdir -p "logs/$BENCHMARK"

# Run the pipeline and save the log
python3 orchestration_pipeline.py --benchmark "$BENCHMARK" \
    > "logs/$BENCHMARK/$(date +%Y%m%d_%H%M%S).log"
