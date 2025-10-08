import torch
import pytest
from training.configurator import GPTConfig
from training.pretraining.components.transformer import GPTWithMHA


def test_transformer_can_be_built():
    """Test that transformer model can be instantiated."""
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
    )

    model = GPTWithMHA(config)
    assert model is not None
    assert model.config.n_layer == 4
    assert model.config.n_head == 4
    assert model.config.n_embd == 128


def test_transformer_forward_pass():
    """Test that transformer can perform forward pass."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    model = GPTWithMHA(config)
    model.eval()

    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, _ = model(idx)

    assert logits.shape == (B, 1, config.vocab_size)


def test_transformer_forward_with_targets():
    """Test that transformer can compute loss with targets."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    model = GPTWithMHA(config)
    model.eval()

    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    with torch.no_grad():
        logits, loss = model(idx, targets=targets)

    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.item() >= 0
    assert not torch.isnan(loss)


def test_transformer_generate():
    """Test that transformer can generate tokens."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    model = GPTWithMHA(config)
    model.eval()

    B, T = 1, 10
    idx = torch.randint(0, config.vocab_size, (B, T))

    generated = model.generate(idx, max_new_tokens=5, temperature=1.0)

    assert generated.shape[0] == B
    assert generated.shape[1] >= T
    assert generated.shape[1] <= T + 5


def test_transformer_num_params():
    """Test that parameter counting works."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    model = GPTWithMHA(config)
    n_params = model.get_num_params(non_embedding=True)

    assert n_params > 0
    assert isinstance(n_params, int)
