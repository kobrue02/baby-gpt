"""
Configuration settings for training the transformer model.
"""

import torch
import subprocess as sp
import os
import yaml

from dataclasses import dataclass
from pathlib import Path


def get_gpu_memory():
    """Get the available GPU memory in MB."""
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


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
    use_rotary: bool = True  # use rotary embeddings


def _load_yaml_config(config_name: str) -> dict:
    """Load a YAML configuration file from the configs directory."""
    config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device_config(config_type: str = "pretrain"):
    """Detect available device and return optimized settings.

    Args:
        config_type: Type of config to use for batch_size override ("pretrain" or "sft")
    """
    if torch.cuda.is_available():

        gpu_mem = get_gpu_memory()
        n_gpus = len(gpu_mem)
        print(f"Detected {n_gpus} GPU(s) with memory: {gpu_mem} MB")

        if n_gpus > 1:
            raise NotImplementedError(
                "Multi-GPU support is not implemented in this script."
            )
        elif n_gpus == 1:
            gpu_mem = gpu_mem[0]
            if 9000 <= gpu_mem < 15000:
                block_size = 512
                batch_size = 4
                grad_accum_steps = 8
            elif 15000 <= gpu_mem < 17000:
                block_size = 512
                batch_size = 8
                grad_accum_steps = 8
            elif 17000 <= gpu_mem < 25000:
                block_size = 2048
                batch_size = 8
                grad_accum_steps = 8
            elif gpu_mem >= 25000:
                block_size = 2048
                batch_size = 16
                grad_accum_steps = 8
            else:
                block_size = 512
                batch_size = 2
                grad_accum_steps = 16

        # For SFT, override batch_size to 4 (as in original load_sft_config)
        if config_type == "sft":
            batch_size = 4

        return {
            "device": "cuda",
            "dtype": "float16",
            "compile": False,
            "gradient_accumulation_steps": grad_accum_steps,
            "batch_size": batch_size,
            "block_size": block_size,
            "wandb_project": "baby-gpt-cuda",
            "wandb_run_name": "baby-gpt-cuda-run",
        }

    elif torch.backends.mps.is_available():
        # For SFT, use batch_size 4; for pretrain, use 2
        batch_size = 4 if config_type == "sft" else 2

        return {
            "device": "mps",
            "dtype": "float32",
            "compile": False,
            "gradient_accumulation_steps": 4,
            "batch_size": batch_size,
            "block_size": 1024,
            "wandb_project": "baby-gpt-mps",
            "wandb_run_name": "baby-gpt-mps-run",
        }
    else:
        # CPU fallback with conservative settings
        # For SFT, use batch_size 4; for pretrain, use 1
        batch_size = 4 if config_type == "sft" else 1

        return {
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "gradient_accumulation_steps": 16,
            "batch_size": batch_size,
            "block_size": 512,
            "wandb_project": "baby-gpt-cpu",
            "wandb_run_name": "baby-gpt-cpu-run",
        }


def load_config():
    """Load configuration settings for training."""
    # Load static configuration from YAML
    config = _load_yaml_config("pretrain")

    # Get dynamic device-specific configuration
    device_config = get_device_config(config_type="pretrain")

    gradient_accumulation_steps = device_config["gradient_accumulation_steps"]
    batch_size = device_config["batch_size"]
    block_size = device_config["block_size"]

    # Calculate derived values
    ddp_world_size = config["ddp_world_size"]
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )

    # Merge dynamic values with static config
    config.update({
        # Derived/calculated values
        "tokens_per_iter": tokens_per_iter,
        # Dynamic device-specific values
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "device": device_config["device"],
        "dtype": device_config["dtype"],
        "compile": device_config["compile"],
        "wandb_project": device_config["wandb_project"],
        "wandb_run_name": device_config["wandb_run_name"],
    })

    return config


def load_sft_config():
    """Load configuration settings for supervised fine-tuning (SFT)."""
    # Load static configuration from YAML
    config = _load_yaml_config("sft")

    # Get dynamic device-specific configuration (with sft batch_size override)
    device_config = get_device_config(config_type="sft")

    gradient_accumulation_steps = device_config["gradient_accumulation_steps"]
    batch_size = config["batch_size"]  # Use batch_size from YAML (which is 4)
    block_size = device_config["block_size"]

    # Calculate derived values
    ddp_world_size = config["ddp_world_size"]
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )

    # Merge dynamic values with static config
    config.update({
        # Derived/calculated values
        "tokens_per_iter": tokens_per_iter,
        # Dynamic device-specific values
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "block_size": block_size,
        "device": device_config["device"],
        "dtype": device_config["dtype"],
        "compile": device_config["compile"],
        "wandb_project": device_config["wandb_project"].replace(
            "baby-gpt", "baby-gpt-sft"
        ),
        "wandb_run_name": device_config["wandb_run_name"].replace("run", "sft-run"),
    })

    return config
