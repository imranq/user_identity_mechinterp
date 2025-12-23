#!/usr/bin/env bash
set -euo pipefail

git pull --rebase

python3 run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 \
  --device cuda --probe_position question_last_token --align_probe_index \
  | tee latest_probe.txt
