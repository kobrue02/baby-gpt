import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

from training.pretraining.components.yarn import (
    apply_rotary_pos_emb,
    LlamaYaRNScaledRotaryEmbedding,
)


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    training: bool = True,
    temp_mask: Optional[torch.Tensor] = None,

) -> torch.Tensor:
    """
    Scaled Dot-Product Attention with optional causal masking and attention mask.
    Implemented manually for compatibility with PyTorch versions < 2.0.

    Args:
        query (torch.Tensor): query of shape (..., L, E)
        key (torch.Tensor): key of shape (..., S, E)
        value (torch.Tensor): value of shape (..., S, E_v)
        attn_mask (torch.Tensor, optional): attention mask of shape (L, S). Default: None
        dropout_p (float, optional): Dropout probability. Default: 0.0
        is_causal (bool, optional): Whether to apply causal mask. Default: False
        scale (float, optional): Scaling factor for dot product. Default: None
        enable_gqa (bool, optional): Whether to enable Grouped Query Attention (GQA). Default: False

    Returns:
        attn_output (torch.Tensor): output of shape (..., L, E_v)
    """
    assert (is_causal and temp_mask is not None) or (not is_causal), "temp_mask must be provided if is_causal is True"

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal and temp_mask is not None:
        assert (
            attn_mask is None
        ), "You cannot supply an attention mask and also set causal=True"
        attn_bias.masked_fill_(
            temp_mask.to(attn_bias.device).logical_not(), float("-inf")
        )
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=training)
    
    return attn_weight @ value


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """
    sin_cached: torch.Tensor
    cos_cached: torch.Tensor
    causal_mask_cached: torch.Tensor

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
        apply_rotary_emb: bool = True,
        config=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.config = config
        self.nheads = nheads
        self.dropout = dropout
        self.c_attn = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(E_total, E_q, bias=bias, **factory_kwargs)
        
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.embedding_dim = E_total
        self.bias = bias

        self.apply_rotary_emb = apply_rotary_emb

        if self.apply_rotary_emb:
            assert (
                config is not None
            ), "Config must be provided if using rotary embeddings"
            self.head_dim = config.n_embd // config.n_head
            assert self.head_dim == self.E_head, "Head dim must match E_total // nheads"
            self.block_size: int = config.block_size
            self.scale = 1
            self.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=self.block_size,
                scale=self.scale,
                device=device,
            )
            self.precompute_rotary_emb(device=device, dtype=dtype)

    def precompute_causal_mask(self):
        max_seq_len = self.config.block_size if self.config is not None else 2048
        causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril(diagonal=0)
        # Register as buffer so it moves with model to different devices
        self.register_buffer("causal_mask_cached", causal_mask, persistent=False)
    
    def precompute_rotary_emb(self, device=None, dtype=None):
        # Precompute and cache rotary embeddings for all positions
        dummy_tensor = torch.zeros(1, self.block_size, self.head_dim, device=device, dtype=dtype)
        cos, sin = self.rotary_emb(dummy_tensor, seq_len=self.block_size)
        # Register as buffers so they move with the model to different devices
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x,
        attn_mask=None,
        is_causal=False,
    ) -> Float[torch.Tensor, "batch_size sequence_length n_embd"]:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Step 1. Apply input projection and split into q, k, v
        qkv = self.c_attn(x)  # (B, T, 3 * E_total)
        q, k, v = qkv.split(C, dim=2)  # Each is (B, T, E_total)

        # Step 2. Split heads and prepare for SDPA
        # reshape q, k, v to separate by head
        # (B, T, E_total) -> (B, T, nheads, E_head) -> (B, nheads, T, E_head)
        query = q.view(B, T, self.nheads, self.E_head).transpose(1, 2)
        key = k.view(B, T, self.nheads, self.E_head).transpose(1, 2)
        value = v.view(B, T, self.nheads, self.E_head).transpose(1, 2)

        # Apply rotary embeddings
        if self.apply_rotary_emb:
            assert (
                T <= self.block_size
            ), f"Sequence length {T} exceeds block size {self.block_size}"
            cos = self.cos_cached[:, :T, :]
            sin = self.sin_cached[:, :T, :]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        temp_mask = self.causal_mask_cached[:T, :T]
        # Step 3. Run SDPA
        # (B, nheads, T, E_head)
        if torch.__version__ >= "2.0":
            assert hasattr(torch.nn.functional, "scaled_dot_product_attention")
            # use flash attention if available (PyTorch >= 2.0)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=is_causal,
            )
        else:
            # otherwise use manual implementation
            attn_output = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=(self.dropout if self.training else 0.0),
                is_causal=is_causal,
                temp_mask=temp_mask,
                training=self.training,
            )

        # (B, nheads, T, E_head) -> (B, T, nheads, E_head) -> (B, T, E_total)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.embedding_dim)

        # Step 4. Apply output projection
        # (B, T, E_total) -> (B, T, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output
