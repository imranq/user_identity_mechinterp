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
- `SETUP.md`: Environment setup and notes.

## Quick Start
1) Create a venv and install requirements in `SETUP.md`.
2) Run `python run_all.py --experiment probe --n_questions_per_pair 10 --template_holdout`.
3) Use `python run_all.py --experiment patch --pair_id physics` to patch expert -> novice.

Notes:
- Default model is `google/gemma-3-4b-it`. For quick local smoke tests, use `gpt2`.
- This is a starting scaffold, not a fully tuned experiment.
