import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import MainConfig
from analyse.dataset import build_language_dataloaders
from analyse.gemma_scope import GemmaScopeSAE
from training.sae import SAE


def extract_features():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device

    if cfg.layers is not None:
        layers_to_process = cfg.layers
    else:
        layers_to_process = [i for i in range(cfg.num_layers)]

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Using float32 for inference execution
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, dtype=torch.float32
    )
    model.eval()

    dataloaders = build_language_dataloaders(
        cfg.extract.dataset_path,
        tokenizer,
        batch_size=cfg.extract.batch_size,
        max_length=cfg.extract.max_length,
    )
    languages = list(dataloaders.keys())

    # --- 1. DETERMINE FILENAME ONCE ---
    if cfg.sae_type == "gemma_scope":
        output_filename = "top_features_gemma_scope.json"
        d_sae = 16384  # Or pull dynamically if needed
    elif cfg.sae_type == "custom":
        output_filename = "top_features_custom.json"
        d_sae = cfg.custom_d_sae
    else:
        raise ValueError(f"Unknown sae_type in config: {cfg.sae_type}")

    # --- 2. LOAD EXISTING DATA IF PRESENT ---
    if os.path.exists(output_filename):
        print(f"Found existing {output_filename}. Will update/append new languages.")
        with open(output_filename, "r", encoding="utf-8") as f:
            top_features_dict = json.load(f)
    else:
        print(f"Creating new {output_filename}.")
        top_features_dict = {}

    d_model = model.config.hidden_size

    for layer in layers_to_process:
        print(
            f"\nExtracting features for Layer {layer} using {cfg.sae_type}..."
        )

        layer_name = f"model.layers.{layer}"

        if cfg.sae_type == "gemma_scope":
            sae = GemmaScopeSAE.from_pretrained(
                cfg.sae_repo_id, layer_idx=layer, device=device
            )
        elif cfg.sae_type == "custom":
            sae = SAE.from_pretrained(
                checkpoint_path=cfg.custom_checkpoint_path,
                layer_name=layer_name,
                d_model=d_model,
                d_sae=cfg.custom_d_sae,
                device=device,
            )

        sae.eval()

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

        all_avg_acts = torch.stack(avg_acts_list)

        # Ensure the layer key exists in our dictionary
        layer_key = f"layer_{layer}"
        if layer_key not in top_features_dict:
            top_features_dict[layer_key] = {}

        for idx, lan in enumerate(languages):
            mu_l = all_avg_acts[idx]
            other_acts = torch.cat([all_avg_acts[:idx], all_avg_acts[idx + 1 :]], dim=0)
            mu_other = other_acts.mean(dim=0)
            score = mu_l - mu_other

            top_k_scores, top_k_indices = torch.topk(score, k=cfg.extract.top_k)

            indices = top_k_indices.tolist()
            scores = top_k_scores.tolist()

            u_values = mu_l[top_k_indices].tolist()
            v_values = mu_other[top_k_indices].tolist()

            feature_details = []
            for i in range(len(indices)):
                feature_details.append({
                    "feature_idx": indices[i],
                    "u_target": round(u_values[i], 4),
                    "v_other": round(v_values[i], 4),
                    "score": round(scores[i], 4),
                })

            # --- 3. APPEND/UPDATE THE LANGUAGE ---
            top_features_dict[layer_key][lan] = feature_details

        # --- 4. CHECKPOINT SAVE AFTER EACH LAYER ---
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(top_features_dict, f, indent=4)
        print(f"Checkpoint saved to {output_filename} for {layer_key}.")

    print(f"\nExtraction fully complete. Final data safely in {output_filename}!")


if __name__ == "__main__":
    extract_features()
