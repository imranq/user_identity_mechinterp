# User Identity Mechanistic Interpretability

This folder starts a new experiment inspired by the IOI replication, but focused on how models represent and act on user identity (expert vs novice) using mechanistic techniques.

## Goals
- Locate where user persona representations form (linear probing).
- Test causal importance of persona features (activation patching).
- (Optional) Check chain-of-thought faithfulness with a simple decision pivot analysis.

## Recommended Persona Pairs
- Physics Professor vs 5-Year-Old Child
- Skeptical Auditor vs Gullible Enthusiast
- Formal/Business vs Slang/Informal
- Medical Professional vs Layperson Patient

## File Map
- `persona_prompts.py`: Prompt templates and persona pairs.
- `probe_user_representation.py`: Linear probe for persona representations.
- `activation_patching.py`: Activation patching to test causality.
- `cot_faithfulness.py`: Optional decision pivot / logit lens analysis.
- `run_all.py`: Convenience runner.
- `run_latest.sh`: Pull + run a standard probe with saved output.
- `compute_persona_direction.py`: Compute a persona direction for patching.
- `persona_patching_runner.py`: Apply a persona direction and save outputs.
- `autorater_stub.py`: Prepare outputs for an expert/novice autorater.
- `SETUP.md`: Environment setup and notes.

## Quick Start
1) Create a venv and install requirements in `SETUP.md`.
2) Run a probe:
   `python run_all.py --experiment probe --n_questions_per_pair 10 --template_holdout`
3) Use `run_latest.sh` for a standard probe:
   `bash run_latest.sh`

## User Guide (Probe)
Recommended baseline:
```
python run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 \
  --device cuda:0 --probe_position question_last_token --align_probe_index
```

Controls:
- Remove persona (should go to chance):
  `bash run_latest.sh --drop_persona`
- Shuffle labels (should go to chance):
  `bash run_latest.sh --shuffle_labels`
- Question holdout (stronger generalization test):
  `bash run_latest.sh --question_holdout`

Debugging:
- Show probe token indices:
  `bash run_latest.sh --max_layers 2 --min_layer 1 --show_probe_tokens --show_probe_count 10`
- Show input vectors:
  `bash run_latest.sh --show_embedding_table --show_embedding_table_rows 5 --show_embedding_table_dims 32`

## Persona direction + patching workflow
1) Compute a direction:
   `python compute_persona_direction.py --layer 4 --probe_position question_last_token --align_probe_index`
2) Patch and save outputs:
   `python persona_patching_runner.py --direction_path persona_direction.npy --layer 4 --alpha 3.0 --align_probe_index`
3) Prepare autorater inputs:
   `python autorater_stub.py --input_path patched_outputs.jsonl`
4) Use `python run_all.py --experiment patch --pair_id physics` to patch expert -> novice.

Notes:
- Default model is `google/gemma-3-4b-it`. For quick local smoke tests, use `gpt2`.
- `question_last_token` is the default probe position to avoid persona-token leakage.
- If CUDA device errors occur, pass `--device cuda:0`.
- This is a starting scaffold, not a fully tuned experiment.
