# Setup Notes

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Model notes
- Default model is `google/gemma-3-4b-it` and may require Hugging Face access.
- For quick smoke tests, use `gpt2` with `--model_name gpt2`.

## RunPod setup (Gemma 3 4B-IT)
This follows the same flow as `ioi_replication/RUNPOD_INSTRUCTIONS.md`, adjusted for this experiment.

### Prerequisites
- RunPod account with an API key.
- Your SSH public key added in RunPod account settings.
- `runpodctl` installed and available in `PATH`.

### Create a pod
Use a GPU with at least 24GB VRAM (1x L40S or 1x A100 40GB) and a PyTorch template.

```bash
chmod +x mats/user_identity_mechinterp/runpod_user_identity.sh
export RUNPOD_API_KEY="YOUR_API_KEY"
export RUNPOD_GPU_TYPE="NVIDIA GeForce RTX 4090"
export RUNPOD_POD_NAME="user-identity-mechinterp"
export RUNPOD_IMAGE_NAME="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
export RUNPOD_CONTAINER_DISK_GB=50
export RUNPOD_VOLUME_GB=50
export RUNPOD_MIN_VCPU=8
export RUNPOD_MIN_MEM_GB=30
# Optional: pass tokens into the pod at creation time
export RUNPOD_ENV="HF_TOKEN=...,GITHUB_TOKEN=..."
mats/user_identity_mechinterp/runpod_user_identity.sh create
```

### Connect
```bash
mats/user_identity_mechinterp/runpod_user_identity.sh ssh
```

### Upload or clone the repo
Option A: clone from your Git remote inside the pod.

```bash
git clone <YOUR_REPO_URL>
cd user_identity_mechinterp
```

Option B: rsync from your local machine after the pod is running.

```bash
rsync -av --progress -e "ssh -p <SSH_PORT>" \
  <LOCAL_REPO_PATH>/ \
  root@<SSH_HOST>:/root/user_identity_mechinterp/
```

### Install deps and login to Hugging Face
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install transformer_lens nnsight jaxtyping einops
huggingface-cli login --token "$HF_TOKEN"
```

### Run experiments
```bash
python run_all.py --experiment probe --model_name google/gemma-3-4b-it
python run_all.py --experiment patch --model_name google/gemma-3-4b-it --pair_id physics
```

### Tear down
```bash
mats/user_identity_mechinterp/runpod_user_identity.sh terminate
```

## Git auth on the pod (avoid password prompts)
Use SSH (recommended):
```bash
ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub
# add the key to GitHub, then:
git remote set-url origin git@github.com:imranq/user_identity_mechinterp.git
```

Or cache HTTPS credentials for 24 hours:
```bash
git config --global credential.helper 'cache --timeout=86400'
```

## Research brief: Persona-Reasoning Bottleneck (Gemma 3)
**Question:** When Gemma 3 is prompted with a specific persona (e.g., "5-year-old" vs "PhD"), does it decide on answer complexity/correctness before it starts its Chain-of-Thought?

**Hypothesis:** There is a mid-layer persona direction in the residual stream. Patching this direction from "PhD" into "Child" preserves tone but shifts internal reasoning.

### Experiment A: Logit Lens (Decision point)
- Prompt pair: "You are a PhD in Physics. Explain the twin paradox using Chain-of-Thought." vs "You are a 5-year-old..."
- Run logit lens on the last prompt token (before generation).
- Look for early-layer shifts (e.g., "Lorentz"/"Relativity" vs "Space"/"Fast").

### Experiment B: Activation/Path Patching (Persona circuit)
- Patch mid-layer activations from "PhD" into "Child".
- Metric: logit diff between a simple token (e.g., "fast") and a complex token (e.g., "dilation").
- Identify "persona mover" heads that cause expert concepts to appear.

### 20-hour workflow
1) Setup and verify Gemma 3 in TransformerLens.
2) Linear probe for persona classification on residual stream.
3) Patching to find 3-5 causal heads.
4) Ablate those heads and measure persona coherence.

### Report structure
1) Objective (one sentence).
2) Model & dataset (Gemma 3 4B-IT + persona prompts).
3) Key results (include logit diff plot).
4) Evidence of circuitry (head indices and hypothesized roles).
5) Failures & negative results (explicitly documented).

### Executive summary tips
- 200-300 words, lead with the hook and concrete circuit results.
- Use "unfaithful reasoning" or "circuit failure" instead of "hallucination."

## Quick runs
```bash
python run_all.py --experiment probe --model_name gpt2 --n_questions_per_pair 10 --template_holdout --max_layers 12
python run_all.py --experiment all --model_name gpt2 --n_questions_per_pair 10 --template_holdout --max_layers 12 --reuse_model
```

## Persona direction + patching workflow
```bash
python compute_persona_direction.py --layer 4 --probe_position question_last_token --align_probe_index
python persona_patching_runner.py --direction_path persona_direction.npy --layer 4 --alpha 3.0 --align_probe_index
python autorater_stub.py --input_path patched_outputs.jsonl
python run_all.py --experiment patch --model_name gpt2 --pair_id physics
```
