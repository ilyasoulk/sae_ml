import os
import json
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MainConfig
from analyse.dataset import BalancedCodeSwitchDataset, get_balanced_collate_fn
from analyse.gemma_scope import GemmaScopeSAE

LAN_DISPLAY_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "th": "Thai", "vi": "Vietnamese",
    "zh": "Chinese", "ar": "Arabic",
}

def balanced_code_switch_experiment():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    
    model_name = cfg.llm_path.split("/")[-1]
    base_save_root = f"results/balanced_code_switch/{model_name}"
    os.makedirs(base_save_root, exist_ok=True)

    # setup Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()

    # top features loading
    with open("top_features_gemma_scope.json", "r", encoding="utf-8") as f:
        top_features = json.load(f)

    layers_to_process = cfg.layers if cfg.layers else list(range(cfg.num_layers))
    
    all_results = {s_lan: {t_lan: {"full": [], "isolated": [], "masked": []} 
                          for t_lan in cfg.code_switch.target_languages} 
                  for s_lan in cfg.code_switch.source_languages}

    # 2. Layer-wise Processing
    for layer_idx in layers_to_process:
        print(f" Processing Layer {layer_idx}...")
        layer_key = f"layer_{layer_idx}"
        
        # Load SAE for the current layer
        sae = GemmaScopeSAE.from_pretrained(cfg.sae_repo_id, layer_idx=layer_idx, device=device)
        sae.eval()

        for ori_lan in cfg.code_switch.source_languages:
            if layer_key not in top_features or ori_lan not in top_features[layer_key]:
                continue
            
            # récupère la feature de langue source (
            prefix_feature_idx = top_features[layer_key][ori_lan][0]["feature_idx"]
            
            # Hook setup pour capturer l'activation de la feature spécifique
            def hook_fn(module, input, output, feat_idx=prefix_feature_idx):
                hidden_states = output[0] if isinstance(output, tuple) else output
                with torch.no_grad():
                    acts = sae.encode(hidden_states.to(torch.float32))
                return acts[:, :, feat_idx]

            handle = model.model.layers[layer_idx].register_forward_hook(
                lambda m, i, o: setattr(model, "current_sae_acts", hook_fn(m, i, o))
            )

            for target_lan in cfg.code_switch.target_languages:
                # nouveau dataset qui assure un réequilibrage 60/40
                dataset = BalancedCodeSwitchDataset(
                    cfg.code_switch.balanced_dataset_path, 
                    ori_lan=ori_lan, 
                    target_lan=target_lan
                )
                
                if len(dataset) == 0:
                    continue

                dataloader = DataLoader(
                    dataset, 
                    batch_size=cfg.code_switch.batch_size, 
                    collate_fn=get_balanced_collate_fn(tokenizer)
                )

                batch_full, batch_iso, batch_masked = [], [], []

                with torch.no_grad():
                    for batch in dataloader:
                        # --- Scenario 1: Full Sentence (60/40) ---
                        ids_full = batch["full_input_ids"].to(device)
                        mask_full = batch["full_attention_mask"].to(device)
                        target_mask = batch["target_mask"].to(device) # Zone Langue B
                        
                        model(ids_full, attention_mask=mask_full)
                        if target_mask.sum() > 0:
                            batch_full.append(model.current_sae_acts[target_mask].mean().item())

                        # --- Scenario 2: Isolated Target Segment ---
                        ids_iso = batch["isolated_input_ids"].to(device)
                        mask_iso = batch["isolated_attention_mask"].to(device)
                        model(ids_iso, attention_mask=mask_iso)
                        # On ignore le BOS (token 0) pour la moyenne
                        m_iso = mask_iso.bool()
                        m_iso[:, 0] = False 
                        if m_iso.sum() > 0:
                            batch_iso.append(model.current_sae_acts[m_iso].mean().item())

                        # --- Scenario 3: Masked Intervention ---
                        # On bloque l'attention sur les 60% de Langue A (sauf BOS)
                        intervention_mask = torch.zeros_like(mask_full)
                        intervention_mask[:, 0] = 1 # Keep BOS
                        intervention_mask[target_mask] = 1 # Keep Target Zone
                        
                        model(ids_full, attention_mask=intervention_mask)
                        if target_mask.sum() > 0:
                            batch_masked.append(model.current_sae_acts[target_mask].mean().item())

                # Stockage des moyennes de la couche
                all_results[ori_lan][target_lan]["full"].append(np.mean(batch_full) if batch_full else 0.0)
                all_results[ori_lan][target_lan]["isolated"].append(np.mean(batch_iso) if batch_iso else 0.0)
                all_results[ori_lan][target_lan]["masked"].append(np.mean(batch_masked) if batch_masked else 0.0)

            handle.remove()
            
        # Nettoyage mémoire RAM/VRAM par couche
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Plotting Final
    for ori_lan, targets in all_results.items():
        save_path = f"{base_save_root}/prefix_{ori_lan}"
        os.makedirs(save_path, exist_ok=True)
        prefix_display = LAN_DISPLAY_NAMES.get(ori_lan, ori_lan.upper())

        for target_lan, data in targets.items():
            if not data["full"]: continue
            
            target_display = LAN_DISPLAY_NAMES.get(target_lan, target_lan.upper())
            plt.figure(figsize=(10, 6))
            
            # Courbes
            plt.plot(layers_to_process, data["full"], label="Balanced (60/40)", marker="o", color="#e74c3c")
            plt.plot(layers_to_process, data["isolated"], label="Isolated Target", marker="s", color="#7f8c8d", linestyle="--")
            plt.plot(layers_to_process, data["masked"], label="Prefix Masked (Intervention)", marker="^", color="#3498db")

            plt.title(f"Mechanism Check (60/40): {prefix_display} Feature on {target_display} Segment", fontweight="bold")
            plt.xlabel("Layer Index", fontweight="bold")
            plt.ylabel("Mean Feature Activation", fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/{target_lan}_balanced_intervention.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    balanced_code_switch_experiment()