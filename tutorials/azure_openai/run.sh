#!/usr/bin/env zsh
set -a
source .env
set +a

python3 llm_trainer.py
