#!/bin/bash

export THINKBRAKE_ROOT=/home/work/sglang_thinkbrake

source .venv/bin/activate

MODEL="all"
CATEGORY="all"
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
NUM_WORKERS=10

python thinkbrake/scripts/rollout.py \
    --model $MODEL \
    --category $CATEGORY \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --num_workers $NUM_WORKERS
