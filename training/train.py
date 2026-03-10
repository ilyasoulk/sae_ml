import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


from sae import SAE
from loss import sae_loss
from config import MainConfig
from utils import SAEDataset, get_collate_fn, ActivationBuffer, HookedActivations

if __name__ == "__main__":
    try:
        cfg = MainConfig.load()
        print(f"Loaded config :\n{cfg}")
    except Exception as e:
        print(f"Config Validation Error: \n{e}")
        exit(1)

    wandb.init(
        project="multilingual-sae-project",
        name=f"layer-{cfg.training.target_layer_name}-exp-{cfg.model.expansion_factor}",
        config=cfg.model_dump(),
    )
    target_layer_name = cfg.training.target_layer_name

    print(f"Loading dataset : {cfg.training.dataset_path}")
    dataset = load_dataset(cfg.training.dataset_path)
    train_dataset = SAEDataset(dataset["train"])

    device = torch.device(cfg.training.device)
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.training.llm_path, torch_dtype=torch.bfloat16
    ).to(device)
    llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.training.llm_path)

    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        cfg.training.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(tokenizer, max_length=cfg.training.max_length),
    )

    d_model = llm.config.hidden_size
    module_dict = dict(llm.named_modules())

    if target_layer_name not in module_dict:
        print(f"Available modules: {list(module_dict.keys())[:10]} ...")
        raise ValueError(f"Module '{target_layer_name}' not found in the LLM.")

    target_layer = module_dict[target_layer_name]
    catcher = HookedActivations(target_layer)
    sae = SAE(d_model=d_model, expansion_factor=cfg.model.expansion_factor).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.training.lr)

    buffer = ActivationBuffer(d_model, max_size=cfg.training.max_size, device=device)

    global_step = 0

    for epoch in range(1, cfg.training.num_epochs + 1):
        for batch in train_loader:
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
                for sae_batch in buffer.drain(batch_size=4096):
                    optimizer.zero_grad()
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
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

                    global_step += 1

    wandb.finish()
