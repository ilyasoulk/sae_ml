import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    expansion_factor: int = Field(gt=0)
    l1_coeff: float = Field(gt=0, lt=1)
    loss_type: Literal["l1", "topk"] = "l1"


class TrainConfig(BaseModel):
    batch_size: int = 4096
    lr: float = 3e-4
    max_length: int = 256
    num_epochs: int = 1
    dataset_path: str
    num_tokens: int
    device: str
    llm_path: str
    target_layer_name: str
    max_size: int


class MainConfig(BaseModel):
    model: ModelConfig
    training: TrainConfig

    @classmethod
    def load(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Usage
if __name__ == "__main__":
    try:
        cfg = MainConfig.load()
    except Exception as e:
        print(f"Config Validation Error: \n{e}")
