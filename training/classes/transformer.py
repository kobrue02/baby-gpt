"""
The actual implementation of the transformer (BabyGPT) mostly by Karpathy.
Uses multi-head attention with the MHA class from attn.py instead of CausalSelfAttention.
And uses the PackedSwiGLUFFN activation from act.py, instead of the standard GELU.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Any


class Transformer(ABC, nn.Module):
    """Base class for Transformer models."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config: Any
        self.transformer: nn.ModuleDict
        self.lm_head: nn.Linear

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        raise NotImplementedError("get_num_params method not implemented")

    def _init_weights(self, module):
        """Initialize weights with a near zero normal distribution, and biases to zero."""
        raise NotImplementedError("_init_weights method not implemented")

    def compute_loss(self, logits, targets):
        """Compute the cross-entropy loss, ignoring padding tokens (-1)."""
        raise NotImplementedError("compute_loss method not implemented")

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model."""
        raise NotImplementedError("forward method not implemented")

    def configure_optimizers(self, *args, **kwargs):
        """Set up the optimizer and learning rate scheduler."""
        raise NotImplementedError("configure_optimizers method not implemented")

    @torch.no_grad()
    def generate(self, *args, **kwargs) -> Any:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        raise NotImplementedError("generate method not implemented")
