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
- `compute_persona_direction.py`: Compute a persona direction for patching.
- `persona_patching_runner.py`: Apply a persona direction and save outputs.
- `autorater_stub.py`: Prepare outputs for an expert/novice autorater.
- `SETUP.md`: Environment setup and notes.

## Quick Start
1) Create a venv and install requirements in `SETUP.md`.
2) Run `python run_all.py --experiment probe --n_questions_per_pair 10 --template_holdout`.
3) For multiple experiments without reloading the model each time, use `--reuse_model` and `--experiment all`.

## Persona direction + patching workflow
1) Compute a direction:
   `python compute_persona_direction.py --layer 4 --probe_position question_last_token --align_probe_index`
2) Patch and save outputs:
   `python persona_patching_runner.py --direction_path persona_direction.npy --layer 4 --alpha 3.0 --align_probe_index`
3) Prepare autorater inputs:
   `python autorater_stub.py --input_path patched_outputs.jsonl`
3) Use `python run_all.py --experiment patch --pair_id physics` to patch expert -> novice.

Notes:
- Default model is `google/gemma-3-4b-it`. For quick local smoke tests, use `gpt2`.
- This is a starting scaffold, not a fully tuned experiment.
