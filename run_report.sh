#!/usr/bin/env bash
set -euo pipefail

# Run the report pipeline with batching for faster activation extraction.
# Requires a working venv and installed requirements.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 \
  --device 0 --probe_position question_last_token --align_probe_index \
  --batch_size 32 | tee report_probe.txt

bash run_latest.sh --drop_persona
bash run_latest.sh --shuffle_labels
bash run_latest.sh --question_holdout

python compute_persona_direction.py --layer 4 --probe_position question_last_token \
  --align_probe_index --method mean

python persona_patching_runner.py --direction_path persona_direction.npy \
  --layer 4 --alpha 3.0 --align_probe_index

python autorater_stub.py --input_path patched_outputs.jsonl
