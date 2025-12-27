#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="Qwen/Qwen3-4B-Thinking-2507"
CATEGORY="gsm8k-val"
THRESHOLD="0.05,0.1,0.25,0.5,1.0,2.0,2.5,5.0"
NUM_WORKERS=16
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
MEM_FRACTION_STATIC=0.8

python thinkbrake/scripts/generate.py \
    --model $MODEL \
    --category $CATEGORY \
    --threshold $THRESHOLD \
    --num_workers $NUM_WORKERS \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --mem_fraction_static $MEM_FRACTION_STATIC
