# User Identity Mechanistic Interpretability

This folder contains experiments on how models represent and act on user identity (expert vs novice) using mechanistic interpretability techniques.

## Goals
- Locate where persona representations form (linear probing).
- Test causal impact of persona directions on outputs (steering/patching).
- Measure reasoning effects with logit-lens A/B preference tests.

## Recommended Persona Pairs
- Physics Professor vs 5-Year-Old Child
- Skeptical Auditor vs Gullible Enthusiast
- Formal/Business vs Slang/Informal
- Medical Professional vs Layperson Patient

## File Map (Core)
- `persona_prompts.py`: Prompt templates + persona pairs.
- `probe_user_representation.py`: Persona probing pipeline.
- `compute_persona_direction.py`: Mean-diff persona direction.
- `persona_steer_demo.py`: Steering sweep + sampling + plots.
- `cot_faithfulness.py`: Logit-lens A/B preference evaluation.
- `cot_faithfulness_sweep.py`: Sweep (single model load) + grid plots.
- `ab_summary.py`: Summarize A/B deltas + stats + markdown table.
- `run_all.py`: Convenience runner for probes.
- `run_latest.sh`: Standard probe run with batch size 32.
- `report_draft.md`: Working report text.
- `reasoning_persona_effects.md`: A/B reasoning results summary.
- `SETUP.md`: Environment setup and RunPod notes.

## Quick Start
1) Create a venv and install requirements (see `SETUP.md`).
2) Run a probe:
   `python run_all.py --experiment probe --model_name google/gemma-2-2b-it --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 --device cuda --probe_position question_last_token --align_probe_index --batch_size 32`
3) Run controls:
   `bash run_latest.sh --drop_persona`
   `bash run_latest.sh --shuffle_labels`
4) Compute a persona direction:
   `python compute_persona_direction.py --layer 4 --probe_position question_last_token --align_probe_index --method mean --batch_size 32 --save_path report_artifacts/persona_direction.npy --meta_path report_artifacts/persona_direction.json`
5) Run a steering sweep:
   `python persona_steer_demo.py --direction_path report_artifacts/persona_direction.npy --layers "16,18,20" --alphas "8.0,12.0,16.0" --max_new_tokens 500 --plot_dir report_artifacts --do_sample --temperature 0.7 --top_p 0.9 --prompts_path steer_prompts.txt --out_path report_artifacts/persona_steer_outputs.jsonl`
6) Run A/B logit-lens sweep:
   `python cot_faithfulness_sweep.py --model_name google/gemma-2-2b-it --device cuda --direction_path report_artifacts/persona_direction.npy --layers "22,24" --alphas "32,64" --report_preference --kl_plot --save_curves --skip_hints --grid_plot --out_dir report_artifacts --tag grid_ab --puzzle_ids "rainbows,twin_paradox,photosynthesis,dna_replication,plate_tectonics,entropy,transistor,antibodies"`
7) Summarize deltas:
   `python ab_summary.py --config nohint_L24_a64p0_grid_ab`

## Notes
- Default model used in experiments: `google/gemma-2-2b-it`.
- `question_last_token` is the default probe position to avoid persona-token leakage.
- Batch size 32 is the default for GPU runs.
