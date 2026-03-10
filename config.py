import yaml
from pydantic import BaseModel, Field
from typing import Literal


# training configurations
class ModelConfig(BaseModel):
    expansion_factor: int = Field(gt=0)
    l1_coeff: float = Field(gt=0, lt=10)
    loss_type: Literal["l1", "topk"] = "l1"


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
    llm_path: str
    dataset_path: str
    target_layer_name: str
    device: str
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
    target_languages: list[str] = ["en", "es", "fr", "ja", "ko", "pt", "th", "vi", "zh", "ar"]
    batch_size: int = 32


class AnalyseConfig(BaseModel):
    llm_path: str
    sae_repo_id: str
    num_layers: int = 26
    layers: list[int] | None = None
    device: str
    extract: ExtractConfig
    code_switch: CodeSwitchConfig


# main configuration
class MainConfig(BaseModel):
    training: TrainingConfig
    analyse: AnalyseConfig

    @classmethod
    def load(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
