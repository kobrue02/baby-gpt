"""
Configuration settings for training the transformer model.
"""

import torch

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    device: str | None = None  # 'cpu', 'cuda', 'mps', or None for default
    attn_pdrop: float = 0.0  # attention dropout


def get_device_config():
    """Detect available device and return optimized settings."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "dtype": "float16",
            "compile": False,
            "gradient_accumulation_steps": 8,
            "batch_size": 8,
            "block_size": 1024,
            "wandb_project": "baby-gpt-cuda",
            "wandb_run_name": "baby-gpt-cuda-run",
        }
    elif torch.backends.mps.is_available():
        return {
            "device": "mps",
            "dtype": "float32",
            "compile": False,
            "gradient_accumulation_steps": 4,
            "batch_size": 2,
            "block_size": 1024,
            "wandb_project": "baby-gpt-mps",
            "wandb_run_name": "baby-gpt-mps-run",
        }
    else:
        # CPU fallback with conservative settings
        return {
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "gradient_accumulation_steps": 16,
            "batch_size": 1,
            "block_size": 512,
            "wandb_project": "baby-gpt-cpu",
            "wandb_run_name": "baby-gpt-cpu-run",
        }


def load_config():
    """Load configuration settings for training."""
    device_config = get_device_config()

    gradient_accumulation_steps = device_config["gradient_accumulation_steps"]
    batch_size = device_config["batch_size"]
    block_size = device_config["block_size"]

    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )

    config = {
        # Process settings
        "master_process": master_process,
        "seed_offset": seed_offset,
        "ddp_world_size": ddp_world_size,
        "tokens_per_iter": tokens_per_iter,
        # I/O settings
        "out_dir": "out",
        "eval_interval": 100,
        "log_interval": 200,
        "eval_iters": 200,
        "eval_only": False,
        "always_save_checkpoint": True,
        "init_from": "scratch",
        # Logging
        "wandb_log": False,
        "wandb_project": device_config["wandb_project"],
        "wandb_run_name": device_config["wandb_run_name"],
        # Training hyperparameters
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "block_size": block_size,
        # Model architecture
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.0,
        "bias": True,
        # Optimizer settings
        "learning_rate": 3e-4,
        "max_iters": 10000,
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        # Learning rate schedule
        "decay_lr": True,
        "warmup_iters": 500,
        "lr_decay_iters": 5000,
        "min_lr": 3e-5,
        # Device settings
        "device": device_config["device"],
        "dtype": device_config["dtype"],
        "compile": device_config["compile"],
    }
    return config

def load_sft_config():
    """Load configuration settings for supervised fine-tuning (SFT)."""
    device_config = get_device_config()

    gradient_accumulation_steps = device_config["gradient_accumulation_steps"]
    batch_size = device_config["batch_size"]
    block_size = device_config["block_size"]

    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )

    config = {
        # Process settings
        "master_process": master_process,
        "seed_offset": seed_offset,
        "ddp_world_size": ddp_world_size,
        "tokens_per_iter": tokens_per_iter,

        # I/O settings
        "out_dir": "out_sft",             # separate output directory
        "eval_interval": 100,             # evaluate a bit more frequently
        "log_interval": 200,
        "eval_iters": 200,
        "eval_only": False,
        "always_save_checkpoint": True,
        "init_from": "resume",            # typically resume from a pretrained ckpt

        # Logging
        "wandb_log": False,               # enable if you want
        "wandb_project": device_config["wandb_project"].replace("baby-gpt", "baby-gpt-sft"),
        "wandb_run_name": device_config["wandb_run_name"].replace("run", "sft-run"),

        # Training hyperparameters
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "block_size": block_size,

        # Model architecture (match pretrained checkpoint)
        "n_layer": 4,
        "n_head": 8,
        "n_embd": 512,
        "dropout": 0.0,
        "bias": True,

        # Optimizer settings
        "learning_rate": 5e-5,     # smaller LR for SFT (pretraining was 3e-4)
        "max_iters": 3000,         # usually fewer steps than pretraining
        "weight_decay": 0.0,       # often 0 for SFT to avoid over-regularizing
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,

        # Learning rate schedule
        "decay_lr": True,
        "warmup_iters": 200,
        "lr_decay_iters": 2500,
        "min_lr": 5e-6,

        # Device settings
        "device": device_config["device"],
        "dtype": device_config["dtype"],
        "compile": device_config["compile"],
    }
    return config
