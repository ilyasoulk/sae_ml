import os
import json
import torch
import gc # Added for system RAM cleaning
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
    
    TOP_K = 5 # we'll check the impact of the 5 top ranked features
    
    # graphs to be generated : synergy and features 2 3 4 5 
    metric_keys = ["synergy"] + [f"feat_{i+1}" for i in range(1, TOP_K)]
    
    source_languages = cfg.code_switch.source_languages 
    target_languages = cfg.code_switch.target_languages

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
    
    # Initialize results matrix on fait sans le masked attention
    all_results = {
        s_lan: {
            t_lan: {
                key: {"full": [], "isolated": []} for key in metric_keys
            } for t_lan in target_languages
        } for s_lan in source_languages
    }

    # 2. Layer-wise Processing
    for layer_idx in layers_to_process:
        print(f"Processing Layer {layer_idx}...")
        layer_key = f"layer_{layer_idx}"
        
        sae = GemmaScopeSAE.from_pretrained(cfg.sae_repo_id, layer_idx=layer_idx, device=device)
        sae.eval()

        for ori_lan in source_languages:
            if layer_key not in top_features or ori_lan not in top_features[layer_key]:
                continue
                
            top_k_features = top_features[layer_key][ori_lan][:TOP_K]
            feature_indices = [f["feature_idx"] for f in top_k_features]
            
            def hook_fn(module, input, output, feat_indices=feature_indices):
                hidden_states = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    acts = sae.encode(hidden_states.to(torch.float32))
                return acts[:, :, feat_indices] 

            handle = model.model.layers[layer_idx].register_forward_hook(
                lambda m, i, o: setattr(model, "current_sae_acts", hook_fn(m, i, o))
            )

            for lan in target_languages:
                try:
                    dataset = CodeSwitchDataset(cfg.code_switch.dataset_path, target_lan=lan, ori_lan=ori_lan)
                    if len(dataset) == 0: continue
                    dataloader = DataLoader(dataset, batch_size=cfg.code_switch.batch_size, 
                                            collate_fn=get_code_switch_collate_fn(tokenizer))
                except Exception:
                    continue

                batch_full = {k: [] for k in metric_keys}
                batch_iso = {k: [] for k in metric_keys}

                with torch.no_grad():
                    for batch in dataloader:
                        # --- Scenario 1: Full ---
                        ids_full = batch["full_input_ids"].to(device)
                        mask_full = batch["full_attention_mask"].to(device)
                        noun_mask = batch["noun_mask"].to(device)
                        
                        model(ids_full, attention_mask=mask_full)
                        if noun_mask.sum() > 0:
                            acts = model.current_sae_acts[noun_mask] 
                            # Calcul Synergie (0 à 5)
                            batch_full["synergy"].append((acts > 0.0).float().sum(dim=-1).mean().item())
                            # Calcul Features Individuelles (2, 3, 4, 5)
                            for i in range(1, TOP_K):
                                batch_full[f"feat_{i+1}"].append(acts[:, i].mean().item())

                        # --- Scenario 2: Isolated ---
                        ids_iso = batch["isolated_input_ids"].to(device)
                        mask_iso = batch["isolated_attention_mask"].to(device)
                        
                        model(ids_iso, attention_mask=mask_iso)
                        m_iso = mask_iso.bool()
                        m_iso[:, 0] = False 
                        if m_iso.sum() > 0:
                            acts = model.current_sae_acts[m_iso]
                            batch_iso["synergy"].append((acts > 0.0).float().sum(dim=-1).mean().item())
                            for i in range(1, TOP_K):
                                batch_iso[f"feat_{i+1}"].append(acts[:, i].mean().item())

                # saving results
                for k in metric_keys:
                    all_results[ori_lan][lan][k]["full"].append(np.mean(batch_full[k]) if batch_full[k] else 0.0)
                    all_results[ori_lan][lan][k]["isolated"].append(np.mean(batch_iso[k]) if batch_iso[k] else 0.0)

            handle.remove()
            
        #nettoie vram
        del sae
        gc.collect() 
        torch.cuda.empty_cache() 

    # 3. Plotting
    model_name = cfg.llm_path.split("/")[-1]
    
    for ori_lan in source_languages:
        save_root = f"results/code_switch_causal/{model_name}_top{TOP_K}/prefix_{ori_lan}"
        os.makedirs(save_root, exist_ok=True)
        prefix_name = LAN_DISPLAY_NAMES.get(ori_lan, ori_lan.upper())

        for lan in target_languages:
            noun_name = LAN_DISPLAY_NAMES.get(lan, lan.upper())
            
            # Boucle sur les 5 graphes
            for k in metric_keys:
                data = all_results[ori_lan][lan][k]
                if not data["full"]: continue
                
                plt.figure(figsize=(10, 6))
                
                plt.plot(layers_to_process, data["full"], label="Prefix + Noun", marker="o", color="#e74c3c")
                plt.plot(layers_to_process, data["isolated"], label="Isolated Noun", marker="s", color="#7f8c8d", linestyle="--")

                plt.xlabel("Layer Index", fontweight="bold")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                if k == "synergy":
                    plt.title(f"Mechanism Check: {prefix_name} Top-{TOP_K} Synergy on {noun_name} Noun", fontweight="bold")
                    plt.ylabel(f"Active Features Count (0 to {TOP_K})", fontweight="bold")
                    file_name = f"{lan}_synergy.png"
                else:
                    feat_num = k.split("_")[1]
                    plt.title(f"Mechanism Check: {prefix_name} Feature #{feat_num} on {noun_name} Noun", fontweight="bold")
                    plt.ylabel("Mean Feature Activation", fontweight="bold")
                    file_name = f"{lan}_feature_{feat_num}.png"
                
                plt.tight_layout()
                plt.savefig(f"{save_root}/{file_name}", dpi=300)
                plt.close() 

if __name__ == "__main__":
    code_switch_experiment()