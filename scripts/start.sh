#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR" || exit 1

python3 server.py \
    --model-path $PROJECT_DIR/models/mlx-community/Qwen3-ASR-1_7B-8bit \
    --vad-model-path $PROJECT_DIR/models/fsmn-vad-mlx \
    --host 0.0.0.0 \
    --port 8787 \
    --load-on-startup
