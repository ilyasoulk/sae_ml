import json
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MainConfig
from analyse.dataset import build_language_dataloaders
from analyse.gemma_scope import GemmaScopeSAE


def extract_features():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    layers_to_process = cfg.layers if cfg.layers else list(range(cfg.num_layers))

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, dtype=torch.bfloat16
    )
    model.eval()

    dataloaders = build_language_dataloaders(
        cfg.extract.dataset_path,
        tokenizer,
        batch_size=cfg.extract.batch_size,
        max_length=cfg.extract.max_length,
    )
    languages = list(dataloaders.keys())
    top_features_dict = defaultdict(dict)

    for layer in layers_to_process:
        print(f"Extracting features for Layer {layer}...")

        sae = GemmaScopeSAE.from_pretrained(
            cfg.sae_repo_id, layer_idx=layer, device=device
        )
        sae.eval()

        d_sae = sae.d_sae
        sum_acts = {
            lan: torch.zeros(d_sae, device=device, dtype=torch.float32)
            for lan in languages
        }
        token_counts = {lan: 0 for lan in languages}

        def hook_fn(module, input, output):
            with torch.no_grad():
                hidden_states = output[0] if isinstance(output, tuple) else output
                return sae.encode(hidden_states.to(torch.float32))

        handle = model.model.layers[layer].register_forward_hook(
            lambda m, i, o: setattr(model, "current_sae_acts", hook_fn(m, i, o))
        )

        with torch.no_grad():
            for lan, loader in dataloaders.items():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    feature_mask = attention_mask.clone().bool()
                    feature_mask[:, 0] = False

                    model(input_ids, attention_mask=attention_mask, use_cache=False)
                    acts = model.current_sae_acts

                    acts_flat = acts.view(-1, d_sae)
                    mask_flat = feature_mask.view(-1)
                    valid_acts = acts_flat[mask_flat]

                    sum_acts[lan] += valid_acts.sum(dim=0)
                    token_counts[lan] += valid_acts.shape[0]

        handle.remove()
        del sae
        torch.cuda.empty_cache()


        avg_acts_list = []
        for lan in languages:
            mu_l = sum_acts[lan] / max(token_counts[lan], 1)
            avg_acts_list.append(mu_l)
            
        all_avg_acts = torch.stack(avg_acts_list) # Shape: [num_languages, d_sae]


        for idx, lan in enumerate(languages):
            mu_l = all_avg_acts[idx]
            other_acts = torch.cat([all_avg_acts[:idx], all_avg_acts[idx+1:]], dim=0)
            mu_other = other_acts.mean(dim=0) 
            score = mu_l - mu_other
            
            # Get BOTH the indices and the actual score values
            top_k_scores, top_k_indices = torch.topk(score, k=cfg.extract.top_k)
            
            indices = top_k_indices.tolist()
            scores = top_k_scores.tolist()
            
            # Extract the raw u and v values for these specific features
            u_values = mu_l[top_k_indices].tolist()
            v_values = mu_other[top_k_indices].tolist()
            
            feature_details = []
            for i in range(len(indices)):
                feature_details.append({
                    "feature_idx": indices[i],
                    "u_target": round(u_values[i], 4),
                    "v_other": round(v_values[i], 4),
                    "score": round(scores[i], 4)
                })
                
            top_features_dict[f"layer_{layer}"][lan] = feature_details



    with open("top_features.json", "w", encoding="utf-8") as f:
        json.dump(top_features_dict, f, indent=4)

    print("Successfully saved top_features.json!")


if __name__ == "__main__":
    extract_features()

