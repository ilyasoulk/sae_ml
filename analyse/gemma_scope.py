import torch
import numpy as np
import torch.nn as nn
from huggingface_hub import hf_hub_download, HfApi


class GemmaScopeSAE(nn.Module):
    """
    https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.empty(d_sae))

        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.empty(d_model))

        self.threshold = nn.Parameter(torch.empty(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec

        pre_acts = x_cent @ self.W_enc + self.b_enc

        mask = (pre_acts > self.threshold).float()
        acts = torch.nn.functional.relu(pre_acts) * mask

        return acts

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        reconstructed = acts @ self.W_dec + self.b_dec
        return reconstructed, acts


    
    @classmethod
    def from_pretrained(cls, repo_id: str, layer_idx: int, width: str = "16k", target_l0: int = 40, device: str = "cuda"):
        api = HfApi()
        
        try:
            repo_files = api.list_repo_files(repo_id=repo_id)
        except Exception as e:
            raise RuntimeError(f"Failed to access repository {repo_id}: {e}")

        prefix = f"layer_{layer_idx}/width_{width}/average_l0_"
        suffix = "/params.npz"
        
        available_l0s = []
        for file in repo_files:
            if file.startswith(prefix) and file.endswith(suffix):
                l0_str = file[len(prefix):-len(suffix)]
                try:
                    available_l0s.append(int(l0_str))
                except ValueError:
                    continue
                    
        if not available_l0s:
            raise ValueError(f"No SAEs found in {repo_id} for layer {layer_idx} and width {width}")

        best_l0 = min(available_l0s, key=lambda x: abs(x - target_l0))
        print(f"Layer {layer_idx} | Target L0: {target_l0} -> Selected closest L0: {best_l0}")

        filename = f"layer_{layer_idx}/width_{width}/average_l0_{best_l0}/params.npz"
        
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            raise RuntimeError(f"Failed to download SAE weights for layer {layer_idx}: {e}")

        with np.load(local_path) as data:
            state_dict = {
                "W_enc": torch.tensor(data["W_enc"]),
                "b_enc": torch.tensor(data["b_enc"]),
                "W_dec": torch.tensor(data["W_dec"]),
                "b_dec": torch.tensor(data["b_dec"]),
                "threshold": torch.tensor(data["threshold"])
            }

        d_model = state_dict["W_enc"].shape[0]
        d_sae = state_dict["W_enc"].shape[1]

        sae = cls(d_model=d_model, d_sae=d_sae)
        sae.load_state_dict(state_dict, strict=True)
        
        return sae.to(device)