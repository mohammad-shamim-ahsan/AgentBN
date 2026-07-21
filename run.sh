#!/bin/bash

# Get current experiment from settings.py
EXP=$(python3 -c "from config.settings import EXPERIMENT; print(EXPERIMENT)")

# Create experiment-specific log directory
mkdir -p "logs/$EXP"

# Run the pipeline and save the log
python3 orchestration_pipeline.py > "logs/$EXP/$(date +%Y%m%d_%H%M%S).log"
