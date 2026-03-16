import torch
import wandb
import numpy as np
from tqdm import tqdm
from huggingface_hub import HfApi
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

from config import MainConfig

from training.sae import SAE, TrainableGemmaScopeSAE
from training.loss import sae_loss
from training.utils import (
    SAEDataset,
    get_collate_fn,
    ActivationBuffer,
    HookedActivations,
    MultiHookedActivations,
    MultiActivationBuffer,
    export_saes_to_huggingface
)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    try:
        cfg = MainConfig.load().training
        print(f"Loaded config :\n{cfg}")
    except Exception as e:
        print(f"Config Validation Error: \n{e}")
        exit(1)

    # target_layer_name = cfg.target_layer_name
    target_layer_names = cfg.target_layer_names
    wandb.init(
        project="multilingual-sae-project",
        name=f"layer-{'_'.join(target_layer_names)}-exp-{cfg.model.d_sae}",
        config=cfg.model_dump(),
    )

    print(f"Loading dataset : {cfg.dataset_path}")
    dataset = load_dataset(cfg.dataset_path)
    train_dataset = SAEDataset(dataset["train"])

    device = torch.device(cfg.device)
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)

    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        cfg.optim.llm_batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(tokenizer, max_length=cfg.optim.max_length),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    d_model = llm.config.hidden_size
    print(f"LLM has hidden_size = {d_model}")
    module_dict = dict(llm.named_modules())
    target_modules = {}
    for name in target_layer_names:
        if name not in module_dict:
            print(f"Available modules: {list(module_dict.keys())[:10]} ...")
            raise ValueError(f"Module '{name}' not found in the LLM.")
        target_modules[name] = module_dict[name]

    catcher = MultiHookedActivations(target_modules)
    saes = torch.nn.ModuleDict({
        name.replace(".", "_"): torch.compile( # type: ignore
            SAE(d_model=d_model, d_sae=cfg.model.d_sae)
        )
        for name in target_layer_names
    }).to(device)

    optimizer = torch.optim.AdamW(
        saes.parameters(),
        lr=cfg.optim.lr,
        fused=True,
        weight_decay=cfg.optim.weight_decay,
    )

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.optim.num_warmup_steps
    )

    buffer = MultiActivationBuffer(
        target_layer_names, d_model=d_model, max_size=cfg.optim.max_size, device=device
    )

    global_step = 0

    for epoch in range(1, cfg.optim.num_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                _ = llm.model(input_ids, attention_mask=attention_mask)

            real_acts_dict = {}
            for name in target_layer_names:
                real_acts_dict[name] = catcher.activations[name][attention_mask.bool()]

            buffer.add(real_acts_dict)
            catcher.clear()

            if buffer.is_full:
                saes.train()
                for sae_batch_dict in buffer.drain(batch_size=cfg.optim.sae_batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    metrics_to_log = {}

                    for i, name in enumerate(target_layer_names):
                        safe_name = name.replace(".", "_")
                        l1_coeff = cfg.model.l1_coeff[i]
                        sae = saes[safe_name]
                        # layer_acts = sae_batch_dict[name]
                        layer_acts = sae_batch_dict[name].to(torch.float32)
                        reconstructed_acts, features, mask = sae(layer_acts)
                        layer_loss = sae_loss(
                            layer_acts,
                            reconstructed_acts,
                            features,
                            loss_type=cfg.model.loss_type,
                            ste_mask=mask,
                            l1_coeff=l1_coeff,
                        )
                        total_loss += layer_loss
                        with torch.no_grad():
                            l0 = (features > 0).float().sum(dim=-1).mean().item()
                            mse = (
                                (reconstructed_acts - layer_acts)
                                .pow(2)
                                .sum(dim=-1)
                                .mean()
                                .item()
                            )
                            variance = layer_acts.var(dim=0).sum().item()
                            fve = 1.0 - (mse / (variance + 1e-8))

                            metrics_to_log[f"train/{safe_name}_loss"] = (
                                layer_loss.item()
                            )
                            metrics_to_log[f"train/{safe_name}_l0"] = l0
                            metrics_to_log[f"train/{safe_name}_fve"] = fve

                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if cfg.model.loss_type == "l1":
                        with torch.no_grad():
                            for sae in saes.values():
                                sae.normalize_decoder_weights()

                    metrics_to_log["train/lr"] = scheduler.get_last_lr()[0]
                    wandb.log(metrics_to_log, step=global_step)

                    global_step += 1
                    avg_l0 = sum(metrics_to_log[f"train/{name.replace('.', '_')}_l0"] for name in target_layer_names) / len(target_layer_names)
                    avg_fve = sum(metrics_to_log[f"train/{name.replace('.', '_')}_fve"] for name in target_layer_names) / len(target_layer_names)
                    
                    pbar.set_postfix({
                        "Loss": f"{total_loss.item():.2f}",
                        "Avg_L0": f"{avg_l0:.1f}",
                        "Avg_FVE": f"{avg_fve:.3f}"
                    })

    save_dir = Path("checkpoints") / str(wandb.run.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saes.state_dict(), save_dir / "sae_weights.pt")
    with open(save_dir / "config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=4))

    # if cfg.repo_id:
    #     export_saes_to_huggingface(
    #         saes=saes,
    #         target_layer_names=target_layer_names,
    #         d_sae=cfg.model.d_sae,
    #         metrics=metrics_to_log, # Pass the last batch's metrics dict
    #         repo_id=cfg.repo_id
    #     )
    print(f"Model and config saved to {save_dir}")
    wandb.finish()
