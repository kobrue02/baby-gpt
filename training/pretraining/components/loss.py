import torch
import torch.nn.functional as F
from enum import Enum


class GoldfishStrategy(str, Enum):
    """Enumeration of Goldfish training strategies."""

    STATIC = "static"
    RANDOM = "random"


def compute_goldfish_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    strategy: GoldfishStrategy = GoldfishStrategy.STATIC,
    drop_frequency: int = 4,
    hash_context_width: int = 10,
) -> torch.Tensor:
    """
    Compute the Goldfish loss for language model training.

    Args:
        logits: Tensor of shape (B, T, C)
        targets: Tensor of shape (B, T)
        strategy: Goldfish strategy (STATIC or RANDOM)
        drop_frequency: k parameter for dropping tokens
        hash_context_width: context width for hashing

    Returns:
        scalar loss
    """
    B, T, C = logits.size()
    k = drop_frequency
    h = hash_context_width

    assert strategy in GoldfishStrategy, f"Invalid strategy: {strategy}"
    assert logits.shape[0] == targets.shape[0], "Batch sizes must match"
    assert logits.ndim == 3, "Logits should be 3D (B, T, C)"
    assert k > 1, "Drop frequency k must be greater than 1"
    assert h >= 7, "Hash context width must be >= 7"

    # Initialize mask: True = keep, False = drop
    mask = torch.ones((B, T), device=logits.device, dtype=torch.bool)

    if strategy == GoldfishStrategy.STATIC:
        # Drop every k-th token
        mask[:, k - 1 :: k] = 0

    elif strategy == GoldfishStrategy.RANDOM and T > h:
        raise NotImplementedError("RANDOM strategy is not implemented yet.")

    # Flatten mask and logits for cross_entropy
    logits_flat = logits.view(-1, C)
    targets_flat = targets.view(-1)

    # Apply mask and compute loss
    return F.cross_entropy(
        logits_flat[mask.view(-1)], targets_flat[mask.view(-1)], ignore_index=-1
    )
