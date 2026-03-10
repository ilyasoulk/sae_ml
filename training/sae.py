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

        return x_dec, features
