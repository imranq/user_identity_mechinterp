# Report Steps (MATS-Ready)

This file is a checklist and runbook for producing a strong MATS 10.0 application report using this repo.

## 1) Baseline Probe (Signal Check)
Goal: show persona signal is linearly decodable at the question end.

```bash
python run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 \
  --device cuda:0 --probe_position question_last_token --align_probe_index
```

Save:
- `probe_results.csv`
- console table (layer, accuracy, balanced_accuracy, AUC)

## 2) Controls (Sanity)
Goal: show the signal disappears when it should.

Remove persona:
```bash
bash run_latest.sh --drop_persona
```

Shuffle labels:
```bash
bash run_latest.sh --shuffle_labels
```

Question holdout (generalization):
```bash
bash run_latest.sh --question_holdout
```

Expected:
- Drop persona and shuffle labels go to ~0.5.
- Question holdout should be lower than template holdout if signal is question-specific.

## 3) Direction Extraction
Goal: extract a persona direction for interventions.

Mean-diff direction:
```bash
python compute_persona_direction.py --layer 4 --probe_position question_last_token \
  --align_probe_index --method mean
```

Optional: probe-weight direction:
```bash
python compute_persona_direction.py --layer 4 --probe_position question_last_token \
  --align_probe_index --method probe
```

Outputs:
- `persona_direction.npy`
- `persona_direction.json`

## 4) Patching Intervention
Goal: show causal effect on outputs.

```bash
python persona_patching_runner.py --direction_path persona_direction.npy \
  --layer 4 --alpha 3.0 --align_probe_index
```

Output:
- `patched_outputs.jsonl`

## 5) Autorater Prep (Expert vs Novice)
Goal: prepare outputs for a binary rating task.

```bash
python autorater_stub.py --input_path patched_outputs.jsonl
```

Output:
- `autorater_inputs.csv` (system prompt asks for EXPERT or NOVICE)

## 6) Plots to Include
- Probe accuracy/AUC vs layer.
- Controls: drop persona vs shuffle labels.
- Intervention effect: % outputs rated EXPERT before vs after patch.

## 7) Executive Summary Structure (1 page)
1) Objective: one sentence on persona representation in Gemma 2B.
2) Key result: where persona is linearly decodable (layer range).
3) Controls: drop persona + label shuffle → chance.
4) Intervention: patching shifts expert/novice ratings.
5) Failure modes: e.g., position leakage without alignment.

## 8) Notes for Neel’s Criteria
- Emphasize mechanistic steps (probe + patching + controls).
- Be explicit about negative results and confounds.
- Avoid the term "hallucination"; use "unfaithful reasoning" or "circuit failure".
