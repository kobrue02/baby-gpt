import torch
import pytest
from unittest.mock import patch
import torch.nn.functional as F


def test_masked_loss_basic():
    """Test that masked loss correctly zeros out prompt tokens."""
    with patch("training.sft.train_sft.SFTTrainer.__init__", lambda self: None):
        from training.sft.train_sft import SFTTrainer

        trainer = SFTTrainer()
        trainer.meta_vocab_size = 100
        trainer.Y = torch.randint(0, 100, (2, 10))
        trainer.M = torch.zeros(2, 10)
        trainer.M[:, 5:] = 1.0  # Only compute loss on last 5 tokens

        logits = torch.randn(2, 10, 100)
        loss = trainer._masked_loss(logits)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert not torch.isnan(loss)


def test_masked_loss_all_masked():
    """Test that fully masked input produces NaN (0/0)."""
    with patch("training.sft.train_sft.SFTTrainer.__init__", lambda self: None):
        from training.sft.train_sft import SFTTrainer

        trainer = SFTTrainer()
        trainer.meta_vocab_size = 100
        trainer.Y = torch.randint(0, 100, (2, 10))
        trainer.M = torch.zeros(2, 10)  # All masked

        logits = torch.randn(2, 10, 100)
        loss = trainer._masked_loss(logits)

        # When all masked, we get nan (0/0)
        assert torch.isnan(loss)


def test_masked_loss_no_masking():
    """Test that no masking equals standard cross entropy."""
    with patch("training.sft.train_sft.SFTTrainer.__init__", lambda self: None):
        from training.sft.train_sft import SFTTrainer

        trainer = SFTTrainer()
        trainer.meta_vocab_size = 100
        trainer.Y = torch.randint(0, 100, (2, 10))
        trainer.M = torch.ones(2, 10)  # No masking

        logits = torch.randn(2, 10, 100)
        masked_loss = trainer._masked_loss(logits)

        # Compare to standard cross entropy
        standard_loss = F.cross_entropy(
            logits.view(-1, 100), trainer.Y.view(-1), reduction="mean"
        )

        assert torch.allclose(masked_loss, standard_loss, rtol=1e-5)
