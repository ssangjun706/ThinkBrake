#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="all"
CATEGORY="bfcl-v1,bfcl-v2"
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
NUM_WORKERS=16
MEM_FRACTION_STATIC=0.8
TRIAL=1

python thinkbrake/scripts/rollout.py \
    --model $MODEL \
    --category $CATEGORY \
    --trial $TRIAL \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --mem_fraction_static $MEM_FRACTION_STATIC \
    --num_workers $NUM_WORKERS
