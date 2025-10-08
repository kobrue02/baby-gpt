import torch
import pytest
from training.pretraining.components.loss import (
    compute_goldfish_loss,
    GoldfishStrategy,
)


def test_goldfish_loss_static():
    """Test that static goldfish loss can be computed on arbitrary logits."""
    B, T, C = 2, 32, 100
    logits = torch.randn(B, T, C)
    targets = torch.randint(0, C, (B, T))

    loss = compute_goldfish_loss(
        logits, targets,
        strategy=GoldfishStrategy.STATIC,
        drop_frequency=4,
        hash_context_width=10
    )

    assert loss.shape == torch.Size([])
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_goldfish_loss_random():
    pass


def test_goldfish_loss_assertions():
    """Test that goldfish loss raises appropriate assertions."""
    B, T, C = 2, 16, 100
    logits = torch.randn(B, T, C)
    targets = torch.randint(0, C, (B, T))

    # Test drop frequency assertion
    with pytest.raises(AssertionError):
        compute_goldfish_loss(logits, targets, drop_frequency=1)

    # Test hash context width assertion
    with pytest.raises(AssertionError):
        compute_goldfish_loss(
            logits, targets,
            strategy=GoldfishStrategy.RANDOM,
            hash_context_width=5
        )

    # Test batch size mismatch
    with pytest.raises(AssertionError):
        compute_goldfish_loss(logits, targets[:1, :])
