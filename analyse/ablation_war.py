import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MainConfig
from analyse.gemma_scope import GemmaScopeSAE
from training.sae import SAE

# All 17 target languages (excluding English 'en')
TARGET_LANGUAGES = [
    "es", "fr", "ja", "ko", "pt", "th", "zh", "vi", "ar", # Paper langs
    "yor", "tam", "pan", "sin", "som", "tel", "guj", "zsm" # Custom Aya langs
]

TOP_K_TO_ABLATE = 1 

def load_texts_by_language(dataset_path: str, max_samples: int = 100) -> dict:
    """Loads texts grouped by language from a parallel corpus JSONL."""
    texts_by_lang = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for lang, text in record.items():
                if lang not in texts_by_lang:
                    texts_by_lang[lang] = []
                if len(texts_by_lang[lang]) < max_samples:
                    texts_by_lang[lang].append(text)
    return texts_by_lang

def compute_batched_ce_loss(model, tokenizer, texts, device, batch_size=16, max_length=128):
    """Computes mean Cross-Entropy loss for a list of texts."""
    if not texts: return 0.0
    
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().bool()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_logits.size(0), -1)

        for j in range(token_losses.size(0)):
            valid_losses = token_losses[j][shift_mask[j]]
            total_loss += valid_losses.sum().item()
            total_tokens += valid_losses.numel()

    return total_loss / max(total_tokens, 1)

def build_ablation_hook(sae, feature_indices):
    """Builds a hook to project out specific SAE features handling both SAE types."""
    if hasattr(sae, "W_dec"):
        decoder_directions = sae.W_dec[feature_indices].to(torch.float32)
    else:
        decoder_directions = sae.dec.weight[:, feature_indices].T.to(torch.float32)

    norms_squared = (decoder_directions**2).sum(dim=1, keepdim=True)
    normalised_directions = decoder_directions / norms_squared
    projection_matrix = (decoder_directions.T @ normalised_directions)

    def hook_fn(module, layer_input, layer_output):
        is_tuple = isinstance(layer_output, tuple)
        hidden_states = layer_output[0] if is_tuple else layer_output
        hidden_states_f32 = hidden_states.to(torch.float32)
        
        ablated = hidden_states_f32 - hidden_states_f32 @ projection_matrix.to(hidden_states_f32.device)
        ablated = ablated.to(hidden_states.dtype)
        
        return (ablated,) + layer_output[1:] if is_tuple else ablated

    return hook_fn

def run_large_ablation():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    layer = cfg.layers[0] if cfg.layers else 20
    layer_key = f"layer_{layer}"
    
    # OVERRIDE the dataset path to the new large one
    dataset_path = "data/parallel_corpus_large.jsonl"
    out_dir = "results/ablation_large"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(cfg.llm_path, device_map=device, torch_dtype=torch.bfloat16)
    model.eval()

    print("Loading Dataset...")
    texts_by_lang = load_texts_by_language(dataset_path, max_samples=100)
    all_langs = list(texts_by_lang.keys())

    # --- 1. Compute Baselines ---
    print("\nComputing Baseline CE Losses...")
    baseline_ce = {}
    for lang in all_langs:
        baseline_ce[lang] = compute_batched_ce_loss(model, tokenizer, texts_by_lang[lang], device, cfg.ablation.batch_size)

    results = {"Gemma Scope": {}, "Custom SAE (Aya)": {}}

    # --- 2. Define SAE Configs ---
    sae_configs = [
        {
            "name": "Gemma Scope",
            "json_file": "top_features_gemma_scope.json",
            "load_fn": lambda: GemmaScopeSAE.from_pretrained(cfg.sae_repo_id, layer_idx=layer, device=device)
        },
        {
            "name": "Custom SAE (Aya)",
            "json_file": "top_features_custom.json",
            "load_fn": lambda: SAE.from_pretrained(cfg.custom_checkpoint_path, layer_name=f"model.layers.{layer}", d_model=model.config.hidden_size, d_sae=cfg.custom_d_sae, device=device)
        }
    ]

    # --- 3. Run Ablations ---
    for sae_cfg in sae_configs:
        sae_name = sae_cfg["name"]
        print(f"\n--- Evaluating {sae_name} ---")
        
        with open(sae_cfg["json_file"], "r") as f:
            top_features = json.load(f)

        sae = sae_cfg["load_fn"]()
        sae.eval()

        for target_lang in TARGET_LANGUAGES:
            if target_lang not in top_features[layer_key] or target_lang not in texts_by_lang:
                results[sae_name][target_lang] = {"target_delta": 0.0, "collateral_delta": 0.0}
                continue
                
            feature_records = top_features[layer_key][target_lang][:TOP_K_TO_ABLATE]
            feature_indices = [rec["feature_idx"] for rec in feature_records]
            
            print(f"Ablating {sae_name} Rank #1 feature for {target_lang.upper()} (Index: {feature_indices})")
            
            hook_fn = build_ablation_hook(sae, feature_indices)
            handle = model.model.layers[layer].register_forward_hook(hook_fn)

            ablated_target_ce = compute_batched_ce_loss(model, tokenizer, texts_by_lang[target_lang], device, cfg.ablation.batch_size)
            target_delta = ablated_target_ce - baseline_ce[target_lang]

            collateral_deltas = []
            for other_lang in all_langs:
                if other_lang != target_lang:
                    ablated_other_ce = compute_batched_ce_loss(model, tokenizer, texts_by_lang[other_lang], device, cfg.ablation.batch_size)
                    collateral_deltas.append(max(0, ablated_other_ce - baseline_ce[other_lang])) # Floor at 0
            
            collateral_delta = sum(collateral_deltas) / len(collateral_deltas)

            handle.remove()
            
            results[sae_name][target_lang] = {
                "target_delta": target_delta,
                "collateral_delta": collateral_delta
            }

        del sae
        torch.cuda.empty_cache()

    # --- 4. Plot Results ---
    print("\nGenerating Large Ablation Plot...")
    
    x = np.arange(len(TARGET_LANGUAGES))
    width = 0.35

    # 2 Rows, 1 Column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    gemma_target = [results["Gemma Scope"].get(l, {}).get("target_delta", 0) for l in TARGET_LANGUAGES]
    custom_target = [results["Custom SAE (Aya)"].get(l, {}).get("target_delta", 0) for l in TARGET_LANGUAGES]
    
    gemma_collat = [results["Gemma Scope"].get(l, {}).get("collateral_delta", 0) for l in TARGET_LANGUAGES]
    custom_collat = [results["Custom SAE (Aya)"].get(l, {}).get("collateral_delta", 0) for l in TARGET_LANGUAGES]

    # Subplot 1: Target Spike
    ax1.bar(x - width/2, gemma_target, width, label='Gemma Scope', color='#e74c3c', edgecolor='black')
    ax1.bar(x + width/2, custom_target, width, label='Custom SAE (Aya)', color='#3498db', edgecolor='black')
    ax1.set_ylabel('$\Delta$ CE Loss on Target', fontweight='bold', fontsize=12)
    ax1.set_title(f'Feature Causality: Target Language Spike (Higher = Better)', fontweight='bold', fontsize=14)
    ax1.legend(frameon=False, fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 2: Collateral Damage
    ax2.bar(x - width/2, gemma_collat, width, label='Gemma Scope', color='#f5b041', edgecolor='black', hatch='//')
    ax2.bar(x + width/2, custom_collat, width, label='Custom SAE (Aya)', color='#85c1e9', edgecolor='black', hatch='//')
    ax2.set_ylabel('Average $\Delta$ CE Loss on Others', fontweight='bold', fontsize=12)
    ax2.set_title(f'Feature Precision: Collateral Damage (Lower = Better)', fontweight='bold', fontsize=14)
    ax2.legend(frameon=False, fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # X-Axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels([lang.upper() for lang in TARGET_LANGUAGES], fontweight='bold', fontsize=11)
    
    # Add visual divider between Paper Langs and Aya Langs
    divider_x = 8.5 # Between AR (index 8) and YOR (index 9)
    for ax in [ax1, ax2]:
        ax.axvline(x=divider_x, color='black', linestyle='-', alpha=0.8, linewidth=2)
        ax.text(divider_x - 0.5, ax.get_ylim()[1]*0.9, 'Base Paper Languages', ha='right', fontweight='bold', alpha=0.6)
        ax.text(divider_x + 0.5, ax.get_ylim()[1]*0.9, 'Custom Aya Languages', ha='left', fontweight='bold', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"layer_{layer}_large_ablation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    run_large_ablation()