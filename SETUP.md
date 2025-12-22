# Setup Notes

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Model notes
- `google/gemma-2-2b` and `meta-llama/Meta-Llama-3-8B` usually require Hugging Face access.
- For quick smoke tests, use `gpt2` with `--model_name gpt2`.

## Quick runs
```bash
python run_all.py --experiment probe --model_name gpt2
python run_all.py --experiment patch --model_name gpt2 --pair_id physics
```
