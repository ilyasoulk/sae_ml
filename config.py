import yaml
from pydantic import BaseModel, Field
from typing import Literal


# training configurations
class ModelConfig(BaseModel):
    d_sae: int = Field(gt=0)
    l1_coeff: list[float] = [5.0]
    loss_type: Literal["l1", "topk", "jumprelu"] = "l1"


class OptimConfig(BaseModel):
    llm_batch_size: int = 32
    sae_batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_warmup_steps: int = 1000
    max_length: int = 512
    num_epochs: int = 1
    max_size: int


class TrainingConfig(BaseModel):
    target_layer_names: list[str] = [
        "model.layers.2",
        "model.layers.12",
        "model.layers.20",
    ]
    llm_path: str
    dataset_path: str
    device: str
    repo_id: str = None
    model: ModelConfig
    optim: OptimConfig


# analyse configurations
class ExtractConfig(BaseModel):
    dataset_path: str
    top_k: int = 5
    batch_size: int = 32
    max_length: int = 128


class CodeSwitchConfig(BaseModel):
    dataset_path: str
    target_languages: list[str] = [
        "en",
        "es",
        "fr",
        "ja",
        "ko",
        "pt",
        "th",
        "vi",
        "zh",
        "ar",
    ]
    or_language: str = "es"
    batch_size: int = 32


class AblationConfig(BaseModel):
    dataset_path: str
    target_languages: list[str] = ["fr", "es", "ko"]
    max_samples_per_language: int = 500
    batch_size: int = 16
    # Each entry is [start_idx, topk]: which ranked features to ablate.
    # [0, 1] = rank-#1 feature only, [1, 1] = rank-#2 only, [0, 2] = rank-#1 and #2 together.
    feature_configs: list[list[int]] = [[0, 1], [1, 1], [0, 2]]


class AnalyseConfig(BaseModel):
    llm_path: str
    sae_repo_id: str
    num_layers: int = 26
    layers: list[int] | None = None
    device: str
    extract: ExtractConfig
    code_switch: CodeSwitchConfig
    ablation: AblationConfig
    sae_type: str = "custom" # "gemma_scope" or "custom"
    
    # For Gemma Scope:
    sae_repo_id: str = "google/gemma-scope-2b-pt-res"
    
    # For Custom SAE:
    custom_checkpoint_path: str = "./checkpoints/your_wandb_run_name/sae_weights.pt"
    custom_d_sae: int = 16384


# main configuration
class MainConfig(BaseModel):
    training: TrainingConfig
    analyse: AnalyseConfig

    @classmethod
    def load(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
