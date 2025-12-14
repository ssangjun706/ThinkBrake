#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="all"
CATEGORY="all"
THRESHOLD="0.25,1.0,2.5"
NUM_WORKERS=24
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
MEM_FRACTION_STATIC=0.65

python thinkbrake/scripts/generate.py \
    --model $MODEL \
    --category $CATEGORY \
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --mem_fraction_static $MEM_FRACTION_STATIC
