#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="Qwen/Qwen3-4B-Thinking-2507"
CATEGORY="tool"

python thinkbrake/scripts/evaluate.py \
    --model $MODEL \
    --category $CATEGORY 