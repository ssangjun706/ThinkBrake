#!/bin/bash

sudo apt update
sudo apt-get install tmux -y

curl -LsSf https://astral.sh/uv/install.sh | sh

uv init . -p 3.12
uv venv -p 3.12

source .venv/bin/activate

uv pip install flashinfer-python
uv pip install sglang

uv pip install math-verify[antlr4_13_2]
uv pip install -e .

uv pip install ipykernel ipywidgets matplotlib seaborn