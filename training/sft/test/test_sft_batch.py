import torch
import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch


def test_get_batch_returns_three_tensors():
    """Test that SFT get_batch returns X, Y, and mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal mock data
        block_size = 32
        num_tokens = 1000

        train_tokens = np.random.randint(0, 100, size=num_tokens, dtype=np.uint16)
        train_masks = np.random.randint(0, 2, size=num_tokens, dtype=np.uint8)

        # Write binary files
        with open(os.path.join(tmpdir, "train_sft.bin"), "wb") as f:
            f.write(train_tokens.tobytes())
        with open(os.path.join(tmpdir, "train_sft_mask.bin"), "wb") as f:
            f.write(train_masks.tobytes())

        # Mock trainer with minimal setup
        with patch("training.sft.train_sft.SFTTrainer.__init__", lambda self: None):
            from training.sft.train_sft import SFTTrainer

            trainer = SFTTrainer()
            trainer.config = {"block_size": block_size, "batch_size": 2}
            trainer.device_type = "cpu"
            trainer._seen_batches = set()

            x, y, m = trainer.get_batch("train", tmpdir)

            # Check shapes
            assert x.shape == (2, block_size)
            assert y.shape == (2, block_size)
            assert m.shape == (2, block_size)

            # Check types
            assert x.dtype == torch.int64
            assert y.dtype == torch.int64
            assert m.dtype == torch.float32


def test_find_unseen_batch_tracks_batches():
    """Test that find_unseen_batch adds to seen_batches set."""
    with patch("training.sft.train_sft.SFTTrainer.__init__", lambda self: None):
        from training.sft.train_sft import SFTTrainer

        trainer = SFTTrainer()
        trainer.config = {"block_size": 32, "batch_size": 2}
        trainer._seen_batches = set()

        data = np.arange(1000, dtype=np.uint16)

        ix = trainer.find_unseen_batch(data)

        assert ix.shape == (2,)
        assert len(trainer._seen_batches) == 1
