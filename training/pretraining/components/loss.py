import torch
import torch.nn.functional as F

from enum import Enum

class GoldfishStrategy(str, Enum):
    """Enumeration of Goldfish training strategies."""
    STATIC = "static"
    RANDOM = "random"


def hash_context(context: torch.Tensor) -> int:
    """A simple hash function for a given context tensor."""
    hash_value = hash(context)
    assert hash_value >= 0, "Hash function returned a negative value"
    return hash_value



def compute_goldfish_loss(
        logits, 
        targets, 
        strategy: GoldfishStrategy = GoldfishStrategy.RANDOM, 
        drop_frequency: int = 4, 
        hash_context_width: int = 10
    ) -> torch.Tensor:
    """
    Compute the Goldfish loss based on the specified strategy.

    Args:
        logits: The model's output logits.
        targets: The ground truth target values.
        strategy: The Goldfish training strategy to use (STATIC or RANDOM).
        drop_frequency: Frequency of dropping tokens in RANDOM strategy.
        hash_context_width: Context width for hashing in RANDOM strategy.

    Returns:
        Computed loss value.
    """
    h = hash_context_width
    k = drop_frequency

    assert strategy in GoldfishStrategy, f"Invalid strategy: {strategy}"
    assert logits.shape[0] == targets.shape[0], "Batch sizes must match"
    assert logits.ndim == 3, "Logits should be 3D (batch_size, sequence_length, num_classes)"
    assert k > 1, "Drop frequency k must be greater than 1"
    assert h >= 7, "Hash context width is too small. "
    "For example, if h=7 is used, the model may never learn to produce the word “Power” "
    "at the end of the phrase “the Los Angeles Department of Water and Power.” " 

    
    B, T, C = logits.size()  # batch size, sequence length, number of classes

    if strategy == GoldfishStrategy.STATIC:
        # Static Goldfish: drop every k-th token
        mask = torch.ones((B, T), device=logits.device, dtype=torch.bool)
        mask[:, k-1::k] = 0  # Drop every k-th token
        mask = mask.unsqueeze(-1).expand(-1, -1, C)  # Expand mask to match logits shape
    
    elif strategy == GoldfishStrategy.RANDOM:
        # Random Goldfish: drop tokens based on hash of preceding context
        mask = torch.ones((B, T), device=logits.device, dtype=torch.bool)
        for i in range(T):
            if i >= h:
                context = targets[:, i-h:i]
                context_hash = hash_context(context)
                if context_hash < 1/k:
                    mask[:, i] = 0  # Drop token if hash is less than 1 over k
        mask[:, :h] = 1  # Always keep the first h tokens
        mask = mask.unsqueeze(-1).expand(-1, -1, C)  # Expand mask to match logits shape
    
    logits = logits[mask].view(-1, C)
    targets = targets[mask].view(-1)
    loss = F.cross_entropy(logits, targets, ignore_index=-1)
    return loss
