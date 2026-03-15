import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class SAE(nn.Module):
    """
    https://transformer-circuits.pub/2023/monosemantic-features
    """

    def __init__(self, d_model: int, expansion_factor: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        up_dim = self.expansion_factor * self.d_model

        self.enc = nn.Linear(self.d_model, up_dim)
        self.dec = nn.Linear(up_dim, self.d_model)

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

    def forward(self, x):
        x_cent = x - self.dec.bias
        features = F.relu(self.enc(x_cent))
        x_dec = self.dec(features)

        return x_dec, features, None # to match jumprelu implem


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
