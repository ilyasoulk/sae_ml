import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class SAE(nn.Module):
    """
    https://transformer-circuits.pub/2023/monosemantic-features
    """

    def __init__(self, d_model: int, d_sae : int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.enc = nn.Linear(self.d_model, self.d_sae)
        self.dec = nn.Linear(self.d_sae, self.d_model)

        nn.init.zeros_(self.enc.bias)

        # Normalize decoder weights immediately upon initialization
        self.normalize_decoder_weights()

    def normalize_decoder_weights(self):
        """
        Forces the dictionary vectors (columns of the decoder weight matrix)
        to have a unit L2 norm.
        """
        with torch.no_grad():
            self.dec.weight.data = F.normalize(self.dec.weight.data, p=2, dim=0)


    def encode(self, x):
        x_cent = x - self.dec.bias
        features = F.relu(self.enc(x_cent))
        return features

    def forward(self, x):
        x_cent = x - self.dec.bias
        features = F.relu(self.enc(x_cent))
        x_dec = self.dec(features)

        return x_dec, features, None # to match jumprelu implem
    


    @classmethod
    def from_pretrained(cls, checkpoint_path: str, layer_name: str, d_model: int, d_sae: int, device: str = "cuda"):
        master_state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        layer_prefix = layer_name.replace(".", "_") + "."
        
        layer_state_dict = {}
        for key, tensor in master_state_dict.items():
            if key.startswith(layer_prefix):
                # 1. Strip the layer prefix (e.g., "model_layers_12.")
                clean_key = key[len(layer_prefix):]
                
                # 2. THE FIX: Strip the torch.compile prefix if it exists
                if clean_key.startswith("_orig_mod."):
                    clean_key = clean_key[len("_orig_mod."):]
                    
                layer_state_dict[clean_key] = tensor
                
        if not layer_state_dict:
            raise ValueError(f"No weights found for {layer_name} in {checkpoint_path}")
            
        sae = cls(d_model=d_model, d_sae=d_sae)
        sae.load_state_dict(layer_state_dict, strict=True)
        
        return sae.to(device)

class TrainableGemmaScopeSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))

        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=torch.float32))

        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        self.b_dec = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

        self.threshold = nn.Parameter(torch.ones(d_sae, dtype=torch.float32) * 0.1)

    def encode(self, x: torch.Tensor):
        x_cent = x - self.b_dec

        pre_acts = x_cent @ self.W_enc + self.b_enc

        hard_mask = (pre_acts > self.threshold).float()

        soft_mask = torch.sigmoid(pre_acts - self.threshold)
        mask = hard_mask.detach() - soft_mask.detach() + soft_mask

        acts = F.relu(pre_acts) * mask
        return acts, mask

    def forward(self, x: torch.Tensor):
        acts, mask = self.encode(x)
        reconstructed = acts @ self.W_dec + self.b_dec
        return reconstructed, acts, mask

    def normalize_decoder_weights(self):
        """Must be called after optimizer.step() to prevent norm cheating"""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)
