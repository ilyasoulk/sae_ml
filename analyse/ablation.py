"""
Feature ablation analysis.

For each target language and each model layer, this module:
  1. Identifies the top-ranked language-specific SAE feature directions from top_features.json.
  2. Registers a hook that projects those directions out of the residual stream (ablation).
  3. Computes the change in cross-entropy (CE) loss across the full multilingual test corpus.

A spike in CE loss that is specific to the target language corpus confirms that the
ablated features are *causally* responsible for processing that language at that layer.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from analyse.gemma_scope import GemmaScopeSAE
from config import AblationConfig, MainConfig


LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ar": "Arabic",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_multilingual_samples(
    dataset_path: str,
    max_samples_per_language: int,
) -> dict[str, list[str]]:
    """
    Read a JSONL file of {"text": ..., "lan": ...} records and return a dict
    mapping language code -> list of up to `max_samples_per_language` texts.
    """
    samples_by_language: dict[str, list[str]] = {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            lan = record["lan"]
            text = record["text"]
            if lan not in samples_by_language:
                samples_by_language[lan] = []
            if len(samples_by_language[lan]) < max_samples_per_language:
                samples_by_language[lan].append(text)

    return samples_by_language


# ---------------------------------------------------------------------------
# Batched CE loss computation
# ---------------------------------------------------------------------------


def compute_batched_ce_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    device: str,
    batch_size: int,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute per-sample cross-entropy loss for a list of texts using batched
    inference with padding.  This is significantly faster than the single-sample
    loop used in the reference implementation.

    Returns a 1-D numpy array of shape (len(texts),) with the mean token-level
    NLL for each sample (padding tokens are excluded from the mean).
    """
    per_sample_losses: list[float] = []

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids: Tensor = encoding["input_ids"].to(device)
        attention_mask: Tensor = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits: Tensor = model(
                input_ids, attention_mask=attention_mask, use_cache=False
            ).logits  # (B, T, V)

        # Shift so that token t predicts token t+1
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)
        shift_mask = attention_mask[:, 1:].contiguous().bool()  # (B, T-1)

        # Token-level CE loss, shape (B, T-1)
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_logits.size(0), -1)

        # Mean over valid (non-padding) tokens for each sample
        for i in range(token_losses.size(0)):
            valid_losses = token_losses[i][shift_mask[i]]
            per_sample_losses.append(
                valid_losses.mean().item() if valid_losses.numel() > 0 else 0.0
            )

    return np.array(per_sample_losses, dtype=np.float32)


# ---------------------------------------------------------------------------
# Ablation hook
# ---------------------------------------------------------------------------


def build_ablation_hook(sae: GemmaScopeSAE, feature_indices: list[int]):
    """
    Build a PyTorch forward hook that removes the contribution of the specified
    SAE feature directions from the residual stream.

    For each feature f with decoder direction d_f (shape: d_model):
        coefficient_f = hidden_state · d_f
        hidden_state  = hidden_state - coefficient_f * d_f / ||d_f||²

    When multiple features are ablated together the projection matrix is
    precomputed once to avoid per-token recomputation during inference.

    The returned hook is compatible with `register_forward_hook` on a
    transformer layer module.
    """
    # decoder_directions: (n_features, d_model)
    decoder_directions = sae.W_dec[feature_indices].to(torch.float32)

    # Precompute the projection matrix P such that ablated = x - x @ P.T
    # P[i] = d_i / ||d_i||²  so that x · d_i * (d_i / ||d_i||²) is correct.
    norms_squared = (decoder_directions**2).sum(dim=1, keepdim=True)  # (n_features, 1)
    normalised_directions = decoder_directions / norms_squared  # (n_features, d_model)

    # projection_matrix: (d_model, d_model) — combines all feature projections
    # ablated_x = x - x @ decoder_directions.T @ normalised_directions
    projection_matrix = (
        decoder_directions.T @ normalised_directions
    )  # (d_model, d_model)

    def ablation_hook(
        module: torch.nn.Module,
        layer_input: tuple,
        layer_output,
    ):
        is_tuple = isinstance(layer_output, tuple)
        hidden_states = layer_output[0] if is_tuple else layer_output

        hidden_states_f32 = hidden_states.to(torch.float32)
        ablated = hidden_states_f32 - hidden_states_f32 @ projection_matrix.to(
            hidden_states_f32.device
        )
        ablated = ablated.to(hidden_states.dtype)

        if is_tuple:
            return (ablated,) + layer_output[1:]
        return ablated

    return ablation_hook


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_ablation_experiment() -> None:
    """
    Entry point for the feature ablation analysis.

    For every target language and every (start_idx, topk) feature configuration,
    runs the model layer-by-layer with the corresponding features ablated and
    saves per-sample CE loss arrays as .npy files.

    Results are saved to:
        results/ablation/{model_name}/{target_language}/
            ori_ce_loss.npy
            sae_ce_loss_all_layers_{start_idx}_{topk}.npy   (one per feature_config)
    """
    cfg: AblationConfig = MainConfig.load("config.yaml").analyse
    ablation_cfg = cfg.ablation
    device = cfg.device
    layers_to_process = cfg.layers if cfg.layers else list(range(cfg.num_layers))
    model_name = cfg.llm_path.split("/")[-1]

    with open("top_features.json", "r", encoding="utf-8") as f:
        top_features: dict = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()
    torch.set_grad_enabled(False)

    samples_by_language = load_multilingual_samples(
        ablation_cfg.dataset_path,
        ablation_cfg.max_samples_per_language,
    )
    all_languages = sorted(samples_by_language.keys())

    # Flatten all texts in a consistent language order for bulk operations.
    # We keep track of per-language slice boundaries so we can index into the
    # flat CE loss array without carrying extra bookkeeping downstream.
    ordered_texts: list[str] = []
    language_slice: dict[str, slice] = {}
    cursor = 0
    for lan in all_languages:
        texts = samples_by_language[lan]
        language_slice[lan] = slice(cursor, cursor + len(texts))
        ordered_texts.extend(texts)
        cursor += len(texts)

    for target_language in ablation_cfg.target_languages:
        print(
            f"\n=== Ablation target: {LANGUAGE_DISPLAY_NAMES.get(target_language, target_language)} ==="
        )

        save_dir = f"results/ablation/{model_name}/{target_language}"
        os.makedirs(save_dir, exist_ok=True)

        # --- Baseline CE loss (no hooks) ---
        baseline_path = os.path.join(save_dir, "ori_ce_loss.npy")
        if os.path.exists(baseline_path):
            print("  Baseline CE loss already computed, loading from disk.")
            baseline_ce_loss = np.load(baseline_path)
        else:
            print("  Computing baseline CE loss...")
            baseline_ce_loss = compute_batched_ce_loss(
                model, tokenizer, ordered_texts, device, ablation_cfg.batch_size
            )
            np.save(baseline_path, baseline_ce_loss)
            print(f"  Saved baseline CE loss -> {baseline_path}")

        # --- Ablation per (start_idx, topk) configuration ---
        for start_idx, topk in ablation_cfg.feature_configs:
            output_filename = f"sae_ce_loss_all_layers_{start_idx}_{topk}.npy"
            output_path = os.path.join(save_dir, output_filename)

            if os.path.exists(output_path):
                print(f"  [{start_idx=}, {topk=}] already exists, skipping.")
                continue

            print(f"  Running ablation: start_idx={start_idx}, topk={topk}")

            # ce_loss_per_layer[layer_idx] = np.ndarray of shape (n_total_texts,)
            ce_loss_per_layer: list[np.ndarray] = []

            for layer in layers_to_process:
                layer_key = f"layer_{layer}"
                if (
                    layer_key not in top_features
                    or target_language not in top_features[layer_key]
                ):
                    print(
                        f"    Layer {layer}: no features found for '{target_language}', inserting baseline."
                    )
                    ce_loss_per_layer.append(baseline_ce_loss.copy())
                    continue

                feature_records = top_features[layer_key][target_language]
                selected_features = feature_records[start_idx : start_idx + topk]
                feature_indices = [rec["feature_idx"] for rec in selected_features]

                print(f"    Layer {layer}: ablating features {feature_indices}")

                sae = GemmaScopeSAE.from_pretrained(
                    cfg.sae_repo_id, layer_idx=layer, device=device
                )
                sae.eval()

                hook_fn = build_ablation_hook(sae, feature_indices)
                hook_handle: RemovableHandle = model.model.layers[
                    layer
                ].register_forward_hook(hook_fn)

                layer_ce_loss = compute_batched_ce_loss(
                    model, tokenizer, ordered_texts, device, ablation_cfg.batch_size
                )

                hook_handle.remove()
                del sae
                torch.cuda.empty_cache()

                ce_loss_per_layer.append(layer_ce_loss)

            # Shape: (n_layers, n_total_texts)
            stacked = np.stack(ce_loss_per_layer, axis=0)
            np.save(output_path, stacked)
            print(f"  Saved ablation CE loss -> {output_path}")

    print(f"\nAblation experiment complete. Results in results/ablation/{model_name}/")


if __name__ == "__main__":
    run_ablation_experiment()
