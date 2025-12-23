#!/usr/bin/env bash
set -euo pipefail

git pull --rebase

extra_args=("$@")
if [ "${#extra_args[@]}" -gt 0 ]; then
  echo "Extra args: ${extra_args[*]}"
fi

python3 run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 \
  --device cuda:0 --probe_position question_last_token --align_probe_index \
  --batch_size 32 \
  "${extra_args[@]}" | tee latest_probe.txt
