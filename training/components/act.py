""" Implementation of the SwiGLU feedforward network with custom hidden dimension scaling. """

import torch
import torch.nn as nn
import torch.nn.functional as F

class PackedSwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)