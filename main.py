"""
Main CLI interface for Baby-GPT training using Typer.

This module provides commands to manage pretraining and SFT workflows.
"""

import typer
from typing_extensions import Annotated
from enum import Enum

app = typer.Typer(
    help="LLM Training CLI - Manage pretraining and supervised fine-tuning"
)

# Create subcommand groups
initialize_app = typer.Typer(help="Initialize datasets")
start_app = typer.Typer(help="Start training from scratch")
resume_app = typer.Typer(help="Resume training from checkpoint")

app.add_typer(initialize_app, name="initialize")
app.add_typer(start_app, name="start")
app.add_typer(resume_app, name="resume")


class TrainingMode(str, Enum):
    """Training mode enumeration."""

    PRETRAINING = "pretraining"
    SFT = "sft"


@initialize_app.command("pretraining")
def initialize_pretraining():
    """
    Initialize pretraining dataset by downloading and processing data.

    This command downloads the specified number of shards from the dataset,
    tokenizes the data, and saves it in binary format for training.
    """
    from data.pre import create_pretraining_dataset

    dataset = "HuggingFaceTB/smollm-corpus"
    subset = "cosmopedia-v2"
    n_items = None

    typer.echo(f"Initializing pretraining {dataset}...")
    create_pretraining_dataset(n_items=n_items, dataset_key=dataset, subset=subset)
    typer.secho(
        "Pretraining dataset initialized successfully!",
        fg=typer.colors.GREEN,
        bold=True,
    )


@initialize_app.command("sft")
def initialize_sft(
    n_rows: Annotated[int, typer.Option(help="Number of rows to load")] = 10000,
):
    """
    Initialize SFT (Supervised Fine-Tuning) dataset.

    This command downloads general knowledge Q&A pairs, processes them
    into the SFT format, and saves the tokenized data for training.
    """
    from data.sft import create_sft_dataset

    typer.echo(f"Initializing SFT dataset with {n_rows} rows...")
    create_sft_dataset(n_rows=n_rows)
    typer.secho(
        "SFT dataset initialized successfully!", fg=typer.colors.GREEN, bold=True
    )


@resume_app.command("pretraining")
def resume_pretraining():
    """
    Resume pretraining from the latest checkpoint.

    This command loads the most recent checkpoint and continues
    pretraining from where it left off.
    """
    from training.pretraining.pretrainer import PreTrainer

    typer.echo("Resuming pretraining from checkpoint...")
    trainer = PreTrainer(resume=True)
    trainer.train()


@start_app.command("pretraining")
def start_pretraining():
    """
    Start pretraining from scratch.

    This command initializes a new model and begins pretraining
    without loading from a checkpoint.
    """
    from training.pretraining.pretrainer import PreTrainer

    typer.echo("Starting pretraining from scratch...")
    trainer = PreTrainer(resume=False)
    trainer.train()


@resume_app.command("sft")
def resume_sft():
    """
    Resume SFT from the latest checkpoint.

    This command loads the most recent SFT checkpoint and continues
    fine-tuning from where it left off.
    """
    from training.sft.train_sft import SFTTrainer

    typer.echo("Resuming SFT from checkpoint...")
    trainer = SFTTrainer(resume=True)
    trainer.train()


@start_app.command("sft")
def start_sft():
    """
    Start SFT from scratch or from a pretrained model.

    This command begins supervised fine-tuning, either from scratch
    or by loading a pretrained checkpoint.
    """
    from training.sft.train_sft import SFTTrainer

    typer.echo("Starting SFT...")
    trainer = SFTTrainer(resume=False)
    trainer.train()


@app.command()
def status():
    """
    Show the current status of training runs.

    Displays information about available checkpoints and recent training runs.
    """
    import os
    from pathlib import Path

    typer.secho("\nTraining Status", fg=typer.colors.CYAN, bold=True)
    typer.echo("=" * 50)

    # Check for pretraining checkpoints
    pretrain_dir = Path("out")
    if pretrain_dir.exists() and (pretrain_dir / "ckpt.pt").exists():
        import torch

        ckpt = torch.load(pretrain_dir / "ckpt.pt", map_location="cpu")
        typer.echo(f"\nPretraining:")
        typer.echo(f"  Checkpoint: {pretrain_dir / 'ckpt.pt'}")
        typer.echo(f"  Iteration: {ckpt.get('iter_num', 'N/A')}")
        typer.echo(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A'):.4f}")
    else:
        typer.echo(f"\nPretraining: No checkpoint found")

    # Check for SFT checkpoints
    sft_dir = Path("out_sft")
    if sft_dir.exists() and (sft_dir / "ckpt.pt").exists():
        import torch

        ckpt = torch.load(sft_dir / "ckpt.pt", map_location="cpu")
        typer.echo(f"\nSFT:")
        typer.echo(f"  Checkpoint: {sft_dir / 'ckpt.pt'}")
        typer.echo(f"  Iteration: {ckpt.get('iter_num', 'N/A')}")
        typer.echo(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A'):.4f}")
    else:
        typer.echo(f"\nSFT: No checkpoint found")

    # Check for datasets
    typer.echo(f"\nDatasets:")
    data_dir = Path("data")
    pretrain_data = list(data_dir.glob("*pretrain*.bin")) if data_dir.exists() else []
    sft_data = list(data_dir.glob("*sft*.bin")) if data_dir.exists() else []

    if pretrain_data:
        typer.echo(f"  Pretraining: {len(pretrain_data)} files found")
    else:
        typer.echo(f"  Pretraining: No data found")

    if sft_data:
        typer.echo(f"  SFT: {len(sft_data)} files found")
    else:
        typer.echo(f"  SFT: No data found")

    typer.echo()


@app.command()
def clean(
    mode: Annotated[
        TrainingMode, typer.Argument(help="Which mode to clean (pretraining or sft)")
    ],
    confirm: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
):
    """
    Clean up checkpoints and data for a specific training mode.

    This will remove checkpoints and optionally data files for the specified mode.
    Use with caution as this operation cannot be undone.
    """
    import shutil
    from pathlib import Path

    if mode == TrainingMode.PRETRAINING:
        out_dir = Path("out")
        data_pattern = "*pretrain*.bin"
    else:
        out_dir = Path("out")
        data_pattern = "*sft*.bin"

    if not confirm:
        typer.confirm(
            f"Are you sure you want to clean {mode.value} checkpoints?", abort=True
        )

    # Remove checkpoint directory
    if out_dir.exists():
        shutil.rmtree(out_dir)
        typer.secho(f"Removed {out_dir}", fg=typer.colors.YELLOW)
    else:
        typer.echo(f"No {mode.value} checkpoint directory found")

    typer.secho(f"Cleaned {mode.value} successfully!", fg=typer.colors.GREEN)


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="The text prompt to generate from")],
    checkpoint: Annotated[str, typer.Option(help="Path to checkpoint (default: out/ckpt.pt for pretrain, out_sft/ckpt.pt for sft)")] = None,  # type: ignore
    use_sft: Annotated[
        bool, typer.Option("--sft", help="Use SFT checkpoint instead of pretrained")
    ] = False,
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate")
    ] = 100,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature (higher = more random)")
    ] = 0.8,
    top_k: Annotated[int, typer.Option(help="Top-k sampling parameter (None for no filtering)")] = None,  # type: ignore
    device: Annotated[
        str, typer.Option(help="Device to run on (cpu, cuda, mps)")
    ] = "cpu",
):
    """
    Generate text from a prompt using a trained model.

    This command loads a checkpoint and generates text completions.
    You can use either a pretrained or SFT checkpoint.
    """
    import torch
    import os
    from pathlib import Path
    from training.pretraining.components.transformer import GPTWithMHA
    from training.pretraining.components.blocks import GPTConfig
    from data.utils import enc

    # Determine checkpoint path
    if checkpoint is None:
        if use_sft:
            checkpoint = "out_sft/ckpt.pt"
        else:
            checkpoint = "out/ckpt.pt"

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        typer.secho(
            f"Error: Checkpoint not found at {checkpoint_path}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt["config"]

    # Initialize model
    model_args = dict(
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        block_size=model_config["block_size"],
        bias=model_config["bias"],
        dropout=0.0,  # Disable dropout for inference
        vocab_size=model_config.get("vocab_size", 50304),
    )
    gptconf = GPTConfig(**model_args)
    model = GPTWithMHA(gptconf)

    # Load model weights
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    typer.secho(f"Model loaded successfully!", fg=typer.colors.GREEN)
    typer.echo(
        f"Generating with temperature={temperature}, max_tokens={max_tokens}, top_k={top_k}\n"
    )

    # Encode prompt
    prompt_ids = enc.encode_ordinary(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate
    typer.secho(f"Prompt: {prompt}", fg=typer.colors.CYAN)
    typer.echo("-" * 50)

    with torch.no_grad():
        output_ids = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode output
    output_text = enc.decode(output_ids[0].tolist())
    typer.secho(output_text, fg=typer.colors.WHITE, bold=True)
    typer.echo("-" * 50)
    typer.secho(
        f"\nGenerated {len(output_ids[0]) - len(prompt_ids)} tokens",
        fg=typer.colors.GREEN,
    )


@app.command()
def interactive(
    checkpoint: Annotated[str, typer.Option(help="Path to checkpoint")] = None,  # type: ignore
    use_sft: Annotated[
        bool, typer.Option("--sft", help="Use SFT checkpoint instead of pretrained")
    ] = False,
    max_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate")
    ] = 100,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.8,
    top_k: Annotated[int, typer.Option(help="Top-k sampling parameter")] = None,  # type: ignore
    device: Annotated[str, typer.Option(help="Device to run on")] = "cpu",
):
    """
    Start an interactive text generation session.

    This command loads a checkpoint and allows you to interactively
    generate text from multiple prompts without reloading the model.
    Type 'quit' or 'exit' to end the session.
    """
    import torch
    from pathlib import Path
    from training.pretraining.components.transformer import GPTWithMHA
    from training.pretraining.components.blocks import GPTConfig
    from inference.util import complete
    from data.utils import enc

    # Determine checkpoint path
    if checkpoint is None:
        if use_sft:
            checkpoint = "out_sft/ckpt.pt"
        else:
            checkpoint = "out/ckpt.pt"

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        typer.secho(
            f"Error: Checkpoint not found at {checkpoint_path}",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config = ckpt["config"]

    # Initialize model
    model_args = dict(
        n_layer=model_config["n_layer"],
        n_head=model_config["n_head"],
        n_embd=model_config["n_embd"],
        block_size=model_config["block_size"],
        bias=model_config["bias"],
        dropout=0.0,
        vocab_size=model_config.get("vocab_size", 50304),
    )
    gptconf = GPTConfig(**model_args)
    model = GPTWithMHA(gptconf)

    # Load model weights
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    typer.secho(f"\nModel loaded successfully!", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        f"Settings: temperature={temperature}, max_tokens={max_tokens}, top_k={top_k}"
    )
    typer.echo(
        f"Type your prompt and press Enter to generate. Type 'quit' or 'exit' to end.\n"
    )

    # Interactive loop
    while True:
        try:
            prompt = typer.prompt(typer.style(">>> ", fg=typer.colors.CYAN, bold=True))

            if prompt.lower() in ["quit", "exit", "q"]:
                typer.secho("\nGoodbye!", fg=typer.colors.GREEN)
                break

            if not prompt.strip():
                continue

            output_text = complete(
                model,
                prompt,
                enc,
                device,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            typer.echo(typer.style(output_text, fg=typer.colors.WHITE, bold=True))
            typer.echo()

        except (KeyboardInterrupt, EOFError):
            typer.secho("\n\nGoodbye!", fg=typer.colors.GREEN)
            break


if __name__ == "__main__":
    app()
