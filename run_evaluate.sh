#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="Qwen/Qwen3-4B-Thinking-2507"
CATEGORY="bfcl-v1,bfcl-v2"

python thinkbrake/scripts/evaluate.py \
    --model $MODEL \
    --category $CATEGORY 