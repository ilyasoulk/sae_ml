import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MainConfig
from analyse.dataset import CodeSwitchDataset, get_code_switch_collate_fn
from analyse.gemma_scope import GemmaScopeSAE

lan_dict = {
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


def code_switch_experiment():
    cfg = MainConfig.load("config.yaml").analyse
    device = cfg.device
    target_languages = cfg.code_switch.target_languages

    with open("top_features.json", "r", encoding="utf-8") as f:
        top_features = json.load(f)

    layers_to_process = cfg.layers if cfg.layers else list(range(cfg.num_layers))

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, device_map=device, torch_dtype=torch.bfloat16
    )
    model.eval()

    # Setup Dataloaders for ALL languages upfront
    dataloaders = {}
    datasets = {}
    for lan in target_languages:
        dataset = CodeSwitchDataset(cfg.code_switch.dataset_path, lan)
        datasets[lan] = dataset
        dataloaders[lan] = DataLoader(
            dataset,
            batch_size=cfg.code_switch.batch_size,
            collate_fn=get_code_switch_collate_fn(tokenizer),
        )

    # Dictionaries to store layer-wise results for each language
    results_full_sentence = {lan: [] for lan in target_languages}
    results_isolated_noun = {lan: [] for lan in target_languages}

    for layer in layers_to_process:
        print(f"Processing Layer {layer}...")
        layer_key = f"layer_{layer}"

        # Load SAE ONCE per layer
        sae = GemmaScopeSAE.from_pretrained(
            cfg.sae_repo_id, layer_idx=layer, device=device
        )
        sae.eval()

        for lan in target_languages:
            if len(datasets[lan]) == 0:
                continue

            prefix_code = datasets[lan][0].get("ori_lan", "en")

            if layer_key not in top_features or lan not in top_features[layer_key]:
                print(
                    f"Warning: Feature ID for {lan} at {layer_key} not found. Skipping..."
                )
                # Pad with 0 so the plotting arrays stay aligned with the layers list
                results_full_sentence[lan].append(0.0)
                results_isolated_noun[lan].append(0.0)
                continue

            prefix_feature_idx = top_features[layer_key][prefix_code][0]

            layer_full_acts = []
            layer_isolated_acts = []

            def hook_fn(module, input, output):
                with torch.no_grad():
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    sae_acts = sae.encode(hidden_states.to(torch.float32))
                return sae_acts[:, :, prefix_feature_idx]

            handle = model.model.layers[layer].register_forward_hook(
                lambda m, i, o: setattr(model, "current_sae_acts", hook_fn(m, i, o))
            )

            with torch.no_grad():
                for batch in dataloaders[lan]:
                    # --- Pass 1: Full Sentence ---
                    model(
                        batch["full_input_ids"].to(device),
                        attention_mask=batch["full_attention_mask"].to(device),
                        use_cache=False,
                    )
                    acts = model.current_sae_acts
                    noun_mask = batch["noun_mask"].to(device)

                    # Avoid zero-division if mask is empty
                    if noun_mask.sum() > 0:
                        layer_full_acts.append(acts[noun_mask].mean().item())

                    # --- Pass 2: Isolated Noun ---
                    model(
                        batch["isolated_input_ids"].to(device),
                        attention_mask=batch["isolated_attention_mask"].to(device),
                        use_cache=False,
                    )
                    iso_acts = model.current_sae_acts
                    iso_mask = batch["isolated_attention_mask"].to(device).bool()
                    iso_mask[:, 0] = False

                    if iso_mask.sum() > 0:
                        layer_isolated_acts.append(iso_acts[iso_mask].mean().item())

            handle.remove()

            # Aggregate batch results safely
            mean_full = (
                sum(layer_full_acts) / len(layer_full_acts) if layer_full_acts else 0.0
            )
            mean_iso = (
                sum(layer_isolated_acts) / len(layer_isolated_acts)
                if layer_isolated_acts
                else 0.0
            )

            results_full_sentence[lan].append(mean_full)
            results_isolated_noun[lan].append(mean_iso)

        # Cleanup SAE and flush VRAM before moving to the next layer
        del sae
        torch.cuda.empty_cache()

    # ==========================================
    # Saving and Plotting Logic for ALL Languages
    # ==========================================
    model_name = cfg.llm_path.split("/")[-1]

    for lan in target_languages:
        save_dir = f"results/code_switch/{model_name}/{lan}"
        os.makedirs(save_dir, exist_ok=True)

        data_dict = {
            "layers": layers_to_process,
            "full_sentence_acts": results_full_sentence[lan],
            "isolated_noun_acts": results_isolated_noun[lan],
        }

        with open(os.path.join(save_dir, "data.json"), "w") as f:
            json.dump(data_dict, f, indent=4)

        # Dynamically extract language names for the plot
        noun_name = lan_dict.get(lan, lan.upper())

        # Peek at the first item in the dataset to find the prefix language (ori_lan)
        if len(datasets[lan]) > 0:
            prefix_code = datasets[lan][0].get("ori_lan", "Unknown")
            prefix_name = lan_dict.get(prefix_code, prefix_code.upper())
        else:
            prefix_name = "Unknown"

        plt.figure(figsize=(10, 5))

        plt.plot(
            layers_to_process,
            results_full_sentence[lan],
            label=f"Context: {prefix_name} Prefix + {noun_name} Noun",
            marker="o",
            linewidth=2,
        )
        plt.plot(
            layers_to_process,
            results_isolated_noun[lan],
            label=f"No Context: Isolated {noun_name} Noun",
            marker="s",
            linewidth=2,
        )

        plt.title(
            f"{prefix_name} Feature Activation on {noun_name} Noun (Propagation)",
            fontweight="bold",
        )
        plt.xlabel("Layer Index", fontweight="bold")
        plt.ylabel("Mean Activation", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.savefig(
            os.path.join(save_dir, "activation_plot.png"), bbox_inches="tight", dpi=300
        )
        plt.savefig(os.path.join(save_dir, "activation_plot.pdf"), bbox_inches="tight")
        plt.close()

    print(
        f"Experiment complete! All results saved in results/code_switch/{model_name}/"
    )


if __name__ == "__main__":
    code_switch_experiment()

