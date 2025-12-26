#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="all"
CATEGORY="tool"
NUM_WORKERS=16
MAX_TOTAL_TOKENS=32768
MEM_FRACTION_STATIC=0.8

python thinkbrake/scripts/generate_oracle.py \
    --model $MODEL \
    --category $CATEGORY \
    --num_workers $NUM_WORKERS \
    --max_total_tokens $MAX_TOTAL_TOKENS \
    --mem_fraction_static $MEM_FRACTION_STATIC
