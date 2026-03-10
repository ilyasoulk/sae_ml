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

                    model(input_ids, attention_mask=attention_mask, use_cache=False)
                    acts = model.current_sae_acts

                    acts_flat = acts.view(-1, d_sae)
                    mask_flat = attention_mask.view(-1).bool()
                    valid_acts = acts_flat[mask_flat]

                    sum_acts[lan] += valid_acts.sum(dim=0)
                    token_counts[lan] += valid_acts.shape[0]

        handle.remove()
        del sae
        torch.cuda.empty_cache()

        s_total = torch.stack(list(sum_acts.values())).sum(dim=0)
        n_total = sum(token_counts.values())

        for lan in languages:
            mu_l = sum_acts[lan] / max(token_counts[lan], 1)

            s_other = s_total - sum_acts[lan]
            n_other = n_total - token_counts[lan]
            mu_other = s_other / max(n_other, 1)

            score = mu_l - mu_other
            top_k_indices = torch.topk(score, k=cfg.extract.top_k).indices.tolist()
            top_features_dict[f"layer_{layer}"][lan] = top_k_indices

    with open("top_features.json", "w", encoding="utf-8") as f:
        json.dump(top_features_dict, f, indent=4)

    print("Successfully saved top_features.json!")


if __name__ == "__main__":
    extract_features()

