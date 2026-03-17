import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MainConfig
from analyse.dataset import CodeSwitchDataset, get_code_switch_collate_fn
from torch.utils.data import DataLoader

def run_attention_evolution():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    # CRITICAL: We must use attn_implementation="eager" to get weights!
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" 
    )
    
    # Using Spanish Prefix + French Noun
    dataset = CodeSwitchDataset(cfg.code_switch.dataset_path, target_lan="fr", ori_lan="es")
    loader = DataLoader(dataset, batch_size=1, collate_fn=get_code_switch_collate_fn(tokenizer))
    batch = next(iter(loader))
    
    ids = batch["full_input_ids"].to(device)
    noun_mask = batch["noun_mask"].to(device)
    
    # Index mapping
    noun_indices = noun_mask[0].nonzero(as_tuple=True)[0].tolist()
    # Everything after BOS but before Noun is the "Infection Source" (Prefix)
    prefix_indices = list(range(1, noun_indices[0]))
    bos_index = [0]

    layer_stats = []

    def get_attn_hook(layer_idx):
        def hook(module, input, output):
            # output[1] is the attention weights: (batch, heads, seq, seq)
            if output[1] is None:
                return
            
            # (Heads, S, S) for our single batch item
            attn_weights = output[1][0].detach().cpu().to(torch.float32) 
            
            # Focus only on the Noun tokens' attention rows
            noun_rows = attn_weights[:, noun_indices, :] # (Heads, N_noun_tokens, S)
            
            # Average across heads and tokens to get the "Noun's average focus"
            avg_focus = noun_rows.mean(dim=(0, 1)) # (S,)
            
            # Aggregate based on your mental model
            focus_self = avg_focus[noun_indices].sum().item()
            focus_prefix = avg_focus[prefix_indices].sum().item()
            focus_bos = avg_focus[bos_index].sum().item()
            
            # Normalized
            total = focus_self + focus_prefix + focus_bos
            layer_stats.append({
                "layer": layer_idx,
                "self": focus_self / total,
                "prefix": focus_prefix / total,
                "bos": focus_bos / total
            })
        return hook

    handles = [model.model.layers[i].self_attn.register_forward_hook(get_attn_hook(i)) 
               for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        model(ids, output_attentions=True)

    for h in handles: h.remove()

    # --- Plotting the "Infection" ---
    layers = [s["layer"] for s in layer_stats]
    self_vals = [s["self"] for s in layer_stats]
    prefix_vals = [s["prefix"] for s in layer_stats]
    bos_vals = [s["bos"] for s in layer_stats]

    plt.figure(figsize=(12, 7))
    plt.stackplot(layers, prefix_vals, self_vals, bos_vals, 
                  labels=['Spanish Prefix Focus (Infection)', 'French Noun Focus (Identity)', 'BOS Focus'],
                  colors=['#e74c3c', '#3498db', '#95a5a6'], alpha=0.85)
    
    plt.title("Attention Mixture Evolution: Tracking the 'Infection' of the Noun", fontweight='bold')
    plt.xlabel("Layer Index", fontweight='bold')
    plt.ylabel("Attention Proportion", fontweight='bold')
    plt.axvline(x=20, color='black', linestyle='--', alpha=0.5, label='Explosion Start')
    plt.legend(loc='lower left', frameon=True, facecolor='white')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = "results/attention_infection.png"
    plt.savefig(save_path, dpi=300)
    print(f"Analysis complete. Infection map saved to {save_path}")

if __name__ == "__main__":
    run_attention_evolution()