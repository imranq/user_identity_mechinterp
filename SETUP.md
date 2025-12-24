# Setup Notes

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Model notes
- Default model for the experiments is `google/gemma-2-2b-it`.
- You will need Hugging Face access for Gemma.

## RunPod setup (Gemma-2-2B-IT)
This uses a single RTX 4090 and the PyTorch 2.1.0 image.

### Prerequisites
- RunPod account with an API key.
- Your SSH public key added in RunPod account settings.
- `runpodctl` installed and available in `PATH`.

### Create a pod
Use a GPU with at least 24GB VRAM (1x RTX 4090 works) and a PyTorch template.

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
huggingface-cli login --token "$HF_TOKEN"
```

### Run experiments (core)
```bash
python run_all.py --experiment probe --model_name google/gemma-2-2b-it \
  --n_questions_per_pair 10 --template_holdout --max_layers 12 --min_layer 1 \
  --device cuda --probe_position question_last_token --align_probe_index --batch_size 32

python compute_persona_direction.py --layer 4 --probe_position question_last_token \
  --align_probe_index --method mean --batch_size 32 \
  --save_path report_artifacts/persona_direction.npy \
  --meta_path report_artifacts/persona_direction.json

python persona_steer_demo.py --direction_path report_artifacts/persona_direction.npy \
  --layers "16,18,20" --alphas "8.0,12.0,16.0" \
  --max_new_tokens 500 --plot_dir report_artifacts --do_sample --temperature 0.7 --top_p 0.9 \
  --prompts_path steer_prompts.txt --out_path report_artifacts/persona_steer_outputs.jsonl

python cot_faithfulness_sweep.py --model_name google/gemma-2-2b-it --device cuda \
  --direction_path report_artifacts/persona_direction.npy \
  --layers "22,24" --alphas "32,64" \
  --report_preference --kl_plot --save_curves --skip_hints --grid_plot \
  --out_dir report_artifacts --tag grid_ab \
  --puzzle_ids "rainbows,twin_paradox,photosynthesis,dna_replication,plate_tectonics,entropy,transistor,antibodies"

python ab_summary.py --config nohint_L24_a64p0_grid_ab
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

## Git auth on the pod (avoid password prompts)
Use SSH (recommended):\n```bash\nssh-keygen -t ed25519 -C \"runpod\"\ncat ~/.ssh/id_ed25519.pub\n# add the key to GitHub, then:\ngit remote set-url origin git@github.com:imranq/user_identity_mechinterp.git\n```\n+\n+Or cache HTTPS credentials for 24 hours:\n```bash\ngit config --global credential.helper 'cache --timeout=86400'\n```\n*** End Patch"}}
