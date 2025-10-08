import torch
from training.configurator import GPTConfig
from training.pretraining.components.blocks import Block, MLP, LayerNorm


def test_mlp_forward():
    """Test that MLP can process input."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    mlp = MLP(config)
    B, T, D = 2, 16, config.n_embd
    x = torch.randn(B, T, D)

    output = mlp(x)

    assert output.shape == (B, T, D)
    assert not torch.isnan(output).any()


def test_layer_norm():
    """Test that LayerNorm works correctly."""
    D = 128
    ln = LayerNorm(D, bias=True)

    B, T = 2, 16
    x = torch.randn(B, T, D)

    output = ln(x)

    assert output.shape == (B, T, D)
    assert not torch.isnan(output).any()

    # Check normalization properties
    mean = output.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_block_forward():
    """Test that transformer block can process input."""
    config = GPTConfig(
        block_size=64,
        vocab_size=500,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )

    block = Block(config)
    B, T, D = 2, 16, config.n_embd
    x = torch.randn(B, T, D)

    output = block(x)

    assert output.shape == (B, T, D)
    assert not torch.isnan(output).any()
