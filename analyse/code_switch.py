import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MainConfig
from analyse.dataset import CodeSwitchDataset, get_code_switch_collate_fn
from analyse.gemma_scope import GemmaScopeSAE

LAN_DISPLAY_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "th": "Thai", "vi": "Vietnamese",
    "zh": "Chinese", "ar": "Arabic",
}

def code_switch_experiment():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    target_languages = cfg.code_switch.target_languages
    ori_lan = cfg.code_switch.or_language 

    # 1. Setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()

    with open("top_features_gemma_scope.json", "r", encoding="utf-8") as f:
        top_features = json.load(f)

    layers_to_process = cfg.layers if cfg.layers else list(range(cfg.num_layers))
    results = {lan: {"full": [], "isolated": [], "masked": []} for lan in target_languages}

    # 2. Layer-wise Processing
    for layer_idx in layers_to_process:
        print(f"Processing Layer {layer_idx}...")
        layer_key = f"layer_{layer_idx}"
        
        sae = GemmaScopeSAE.from_pretrained(cfg.sae_repo_id, layer_idx=layer_idx, device=device)
        sae.eval()

        for lan in target_languages:
            if layer_key not in top_features or ori_lan not in top_features[layer_key]:
                continue
                
            prefix_feature_idx = top_features[layer_key][ori_lan][0]["feature_idx"]
            
            dataset = CodeSwitchDataset(cfg.code_switch.dataset_path, target_lan=lan, ori_lan=ori_lan)
            dataloader = DataLoader(dataset, batch_size=cfg.code_switch.batch_size, 
                                    collate_fn=get_code_switch_collate_fn(tokenizer))

            batch_full, batch_iso, batch_masked = [], [], []

            # Hook to capture SAE activations
            def hook_fn(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    acts = sae.encode(hidden_states.to(torch.float32))
                return acts[:, :, prefix_feature_idx]

            handle = model.model.layers[layer_idx].register_forward_hook(
                lambda m, i, o: setattr(model, "current_sae_acts", hook_fn(m, i, o))
            )

            for batch in dataloader:
                # Common data
                ids_full = batch["full_input_ids"].to(device)
                mask_full = batch["full_attention_mask"].to(device)
                noun_mask = batch["noun_mask"].to(device)
                
                # --- Scenario 1: Full Sentence (Standard) ---
                model(ids_full, attention_mask=mask_full)
                if noun_mask.sum() > 0:
                    batch_full.append(model.current_sae_acts[noun_mask].mean().item())

                # --- Scenario 2: Isolated Noun (Control) ---
                ids_iso = batch["isolated_input_ids"].to(device)
                mask_iso = batch["isolated_attention_mask"].to(device)
                model(ids_iso, attention_mask=mask_iso)
                m_iso = mask_iso.bool()
                m_iso[:, 0] = False # Ignore BOS
                if m_iso.sum() > 0:
                    batch_iso.append(model.current_sae_acts[m_iso].mean().item())

                # --- Scenario 3: Causal Masking (The Minimalist Way) ---
                # We tell the model: "The prefix tokens are padding. Ignore them."
                # Nouns only see BOS (index 0) and other Nouns.
                intervention_mask = torch.zeros_like(mask_full)
                intervention_mask[:, 0] = 1        # Keep BOS signal
                intervention_mask[noun_mask] = 1   # Keep Noun signal
                
                model(ids_full, attention_mask=intervention_mask)
                if noun_mask.sum() > 0:
                    batch_masked.append(model.current_sae_acts[noun_mask].mean().item())

            handle.remove()
            
            results[lan]["full"].append(np.mean(batch_full) if batch_full else 0.0)
            results[lan]["isolated"].append(np.mean(batch_iso) if batch_iso else 0.0)
            results[lan]["masked"].append(np.mean(batch_masked) if batch_masked else 0.0)

        del sae
        torch.cuda.empty_cache()

    # 3. Plotting
    model_name = cfg.llm_path.split("/")[-1]
    save_root = f"results/code_switch_causal/{model_name}"
    os.makedirs(save_root, exist_ok=True)
    prefix_name = LAN_DISPLAY_NAMES.get(ori_lan, ori_lan.upper())

    for lan, data in results.items():
        noun_name = LAN_DISPLAY_NAMES.get(lan, lan.upper())
        plt.figure(figsize=(10, 6))
        
        plt.plot(layers_to_process, data["full"], label="Prefix + Noun", marker="o", color="#e74c3c")
        plt.plot(layers_to_process, data["isolated"], label="Isolated Noun", marker="s", color="#7f8c8d", linestyle="--")
        plt.plot(layers_to_process, data["masked"], label="Prefix Masked (Intervention)", marker="^", color="#3498db")

        plt.title(f"Mechanism Check: {prefix_name} Feature on {noun_name} Noun", fontweight="bold")
        plt.xlabel("Layer Index", fontweight="bold")
        plt.ylabel("Mean Feature Activation", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_root}/{lan}_intervention.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    code_switch_experiment()