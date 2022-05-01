#!/usr/bin/env bash

set -m

# Start MlFlow server
mlflow server \
    --host 0.0.0.0 \
    --port 5001 \
    --backend-store-uri sqlite:///.mlruns.db \
    --default-artifact-root .mlruns-artifacts \
    &> .mlflow.log &

# Execute an experiment
python3 src/core/main.py \
    --tracking_uri http://0.0.0.0:5001 \
    --datasets 310 40900 \
    --preprocessings smote tomek_links \
    --total_time 90 \
    --time_per_run 30

echo 'Execute Ctrl+C to terminate...' >&2
fg %1
