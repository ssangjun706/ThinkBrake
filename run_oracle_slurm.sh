#!/bin/bash
#SBATCH --job-name=tb_v2 
#SBATCH --partition=amd_a100nv_8 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8 
#SBATCH --mem=80G #SBATCH --time=0-03:00:00 
#SBATCH -o ./slurm_output/%x_%j.out 
#SBATCH -e ./slurm_output/%x_%j.err 
#SBATCH --comment=pytorch 

set -eo pipefail 

mkdir -p ./slurm_output 


curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

export HF_TOKEN=#
export THINKBRAKE_ROOT=./

uv init . -p 3.12
uv venv -p 3.12

source .venv/bin/activate

uv pip install flashinfer-python
uv pip install sglang

uv pip install math-verify[antlr4_13_2]
uv pip install -e .


MODEL="Qwen/Qwen3-4B-Thinking-2507"
CATEGORY="aime2024"
NUM_WORKERS=24
MAX_TOTAL_TOKENS=32768
MEM_FRACTION_STATIC=0.8

python thinkbrake/scripts/generate_oracle.py \
    --model $MODEL \
    --category $CATEGORY \
    --num_workers $NUM_WORKERS \
    --max_total_tokens $MAX_TOTAL_TOKENS \
    --mem_fraction_static $MEM_FRACTION_STATIC
