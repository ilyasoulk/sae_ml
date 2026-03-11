# sae_ml

Minimal codebase for:

1. Training a Sparse Autoencoder (SAE) on activations from a target LLM layer.
2. Extracting language-specific SAE features across layers.
3. Running a code-switch experiment and plotting activation behavior.

## Repository layout

- `training/train.py`: trains an SAE on hidden activations from one model layer.
- `analyse/extract_features.py`: finds top language-specific SAE features per layer and writes `top_features.json`.
- `analyse/code_switch.py`: uses `top_features.json` to run the code-switch experiment and save plots/data.
- `config.yaml`: main configuration for both training and analysis.

## Requirements

- Python `>=3.12`
- CUDA GPU (strongly recommended)
- Internet access for downloading Hugging Face models/datasets
- Weights & Biases account (training script logs to W&B)

## Installation

### Option A (recommended): `uv`

```bash
uv sync
```

Run scripts with:

```bash
uv run python -m training.train
```

### Option B: `venv + pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Edit `config.yaml`.

Key training fields:

- `training.llm_path`: Hugging Face model ID (example: `Qwen/Qwen2.5-1.5B`)
- `training.dataset_path`: Hugging Face dataset ID (loaded with `datasets.load_dataset`)
- `training.target_layer_name`: module path to hook (example: `model.layers.20`)
- `training.device`: usually `cuda`

Key analysis fields:

- `analyse.llm_path`: model for analysis (example: `google/gemma-2-2b`)
- `analyse.sae_repo_id`: Gemma Scope SAE repo (example: `google/gemma-scope-2b-pt-res`)
- `analyse.num_layers` or `analyse.layers`: layers to process
- `analyse.extract.dataset_path`: JSONL file for multilingual feature extraction
- `analyse.code_switch.dataset_path`: JSONL file for code-switch experiment

## Data format

`analyse.extract.dataset_path` expects JSONL records like:

```json
{ "text": "Some sentence", "lan": "en" }
```

`analyse.code_switch.dataset_path` expects JSONL records like:

```json
{
  "sentence": "Full sentence with switched noun",
  "ori_sentence": "Prefix sentence before noun",
  "ori_lan": "en",
  "target_lan": "es"
}
```

For training data, each dataset item must contain either:

- a `text` field, or
- both `inputs` and `targets` fields (concatenated in code).

## How to run

Run in this order from the repository root:

1. Train SAE:

```bash
uv run python -m training.train
```

Outputs:

- `checkpoints/<wandb-run-name>/sae_weights.pt`
- `checkpoints/<wandb-run-name>/config.json`

2. Extract top language features:

```bash
uv run python -m analyse.extract_features
```

Output:

- `top_features.json`

3. Run code-switch experiment:

```bash
uv run python -m analyse.code_switch
```

Outputs (per target language):

- `results/code_switch/<model_name>/<language>/data.json`
- `results/code_switch/<model_name>/<language>/activation_plot.png`
- `results/code_switch/<model_name>/<language>/activation_plot.pdf`

4. Run ablation experiment:

```bash
uv run python -m analyse.ablation
uv run python -m analyse.plot_ablation
```

Outputs (per target language):

- `results/ablation/<model_name>/<language>/ce_change_<start_idx>_<topk>.png/pdf`
- `results/ablation/<model_name>/<language>/ori_ce_loss.npy`
- `results/ablation/<model_name>/<language>/sae_ce_loss_all_layers_<start_idx>_<topk>.npy`

## Notes

- If training fails with a layer error, verify `training.target_layer_name` exists in `dict(model.named_modules())`.
- If you run out of memory, lower:
  - `training.optim.llm_batch_size`
  - `training.optim.sae_batch_size`
  - `training.optim.max_size`
  - `analyse.extract.batch_size`

```

```
