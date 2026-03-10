import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

from config import MainConfig

from training.sae import SAE
from training.loss import sae_loss
from training.utils import SAEDataset, get_collate_fn, ActivationBuffer, HookedActivations

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    try:
        cfg = MainConfig.load().training
        print(f"Loaded config :\n{cfg}")
    except Exception as e:
        print(f"Config Validation Error: \n{e}")
        exit(1)

    target_layer_name = cfg.target_layer_name
    wandb.init(
        project="multilingual-sae-project",
        name=f"layer-{target_layer_name}-exp-{cfg.model.expansion_factor}",
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

    if target_layer_name not in module_dict:
        print(f"Available modules: {list(module_dict.keys())[:10]} ...")
        raise ValueError(f"Module '{target_layer_name}' not found in the LLM.")

    target_layer = module_dict[target_layer_name]
    catcher = HookedActivations(target_layer)
    sae = SAE(d_model=d_model, expansion_factor=cfg.model.expansion_factor).to(device)
    sae = torch.compile(sae)

    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=cfg.optim.lr,
        fused=True,
        weight_decay=cfg.optim.weight_decay,
    )

    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.optim.num_warmup_steps
    )

    buffer = ActivationBuffer(d_model, max_size=cfg.optim.max_size, device=device)

    global_step = 0

    for epoch in range(1, cfg.optim.num_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                _ = llm(input_ids, attention_mask=attention_mask)

            activation = catcher.activation
            if activation is not None:
                real_acts = activation[attention_mask.bool()]
                buffer.add(real_acts)

            if buffer.is_full:
                sae.train()
                for sae_batch in buffer.drain(batch_size=cfg.optim.sae_batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    reconstructed_acts, features = sae(sae_batch)

                    loss = sae_loss(
                        sae_batch,
                        reconstructed_acts,
                        features,
                        loss_type=cfg.model.loss_type,
                        l1_coeff=cfg.model.l1_coeff,
                    )

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if cfg.model.loss_type == "l1":
                        sae.normalize_decoder_weights()

                    # Interpretability Metrics
                    with torch.no_grad():
                        # L0: Average number of active features per token
                        l0 = (features > 0).float().sum(dim=-1).mean().item()
                        # FVE: Fraction of Variance Explained
                        mse = (
                            (reconstructed_acts - sae_batch)
                            .pow(2)
                            .sum(dim=-1)
                            .mean()
                            .item()
                        )
                        variance = sae_batch.var(dim=0).sum().item()
                        fve = 1.0 - (mse / (variance + 1e-8))
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/l0_sparsity": l0,
                            "train/fve": fve,
                            "train/mse": mse,
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

                    global_step += 1
                    pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "L0": f"{l0:.1f}",
                        "FVE": f"{fve:.3f}",
                    })
    save_dir = Path("checkpoints") / str(wandb.run.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), save_dir / "sae_weights.pt")
    with open(save_dir / "config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=4))

    print(f"Model and config saved to {save_dir}")
    wandb.finish()

