import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from training.components.attn import MultiHeadAttention
from training.components.act import PackedSwiGLUFFN

from dataclasses import dataclass
from jaxtyping import Float, jaxtyped, Int64
from beartype import beartype as typechecker

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str | None = None # 'cpu', 'cuda', 'mps', or None for default
    attn_pdrop: float = 0.0 # attention dropout
    


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.act = PackedSwiGLUFFN(config.n_embd, 4 * config.n_embd, multiple_of=256, device=config.device, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Tensor, "batch_size sequence_length n_embd"]
        ) -> Float[Tensor, "batch_size sequence_length n_embd"]:
        x = self.act(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, input: Float[Tensor, "batch_size sequence_length n_embd"]
        ) -> Float[Tensor, "batch_size sequence_length n_embd"]:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    """ an unassuming Transformer block with MHA """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(
            config.n_embd,
            config.n_embd,
            config.n_embd,
            config.n_embd,
            config.n_head,
            dropout=config.attn_pdrop,
            bias=config.bias,
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Float[Tensor, "batch_size sequence_length n_embd"]
        ) -> Float[Tensor, "batch_size sequence_length n_embd"]:
        q, k, v = self.ln_1(x), self.ln_1(x), self.ln_1(x)
        x = x + self.attn(q, k, v, is_causal=True)
        x = x + self.mlp(self.ln_2(x))
        return x