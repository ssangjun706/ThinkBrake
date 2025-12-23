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

uv pip install ipykernel ipywidgets matplotlib seaborn



MODEL="openai/gpt-oss-20b"
CATEGORY="math500,gsm8k,aime2024,aime2025,gpqa-diamond"
REASONING_TOKENS_BUDGET=16384
ANSWER_TOKENS_BUDGET=4096
NUM_WORKERS=24
MEM_FRACTION_STATIC=0.65

python ./thinkbrake/scripts/rollout.py \
    --model $MODEL \
    --category $CATEGORY \
    --reasoning_tokens_budget $REASONING_TOKENS_BUDGET \
    --answer_tokens_budget $ANSWER_TOKENS_BUDGET \
    --mem_fraction_static $MEM_FRACTION_STATIC \
    --num_workers $NUM_WORKERS


