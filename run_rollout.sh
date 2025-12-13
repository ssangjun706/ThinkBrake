#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="Qwen/Qwen3-32B"
CATEGORY="all"
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
NUM_WORKERS=10
MEM_FRACTION_STATIC=0.6

python thinkbrake/scripts/rollout.py \
    --model $MODEL \
    --category $CATEGORY \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --mem_fraction_static $MEM_FRACTION_STATIC \
    --num_workers $NUM_WORKERS
