#!/bin/bash

export THINKBRAKE_ROOT=/home/work/ThinkBrake

source .venv/bin/activate

MODEL="all"
CATEGORY="all"

python thinkbrake/scripts/evaluate.py \
    --model $MODEL \
    --category $CATEGORY 