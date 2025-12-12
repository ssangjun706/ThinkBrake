#!/bin/bash

export THINKBRAKE_ROOT=/home/work/sglang_thinkbrake

source .venv/bin/activate

MODEL="all"
CATEGORY="all"

python thinkbrake/scripts/evaluate.py \
    --model $MODEL \
    --category $CATEGORY 