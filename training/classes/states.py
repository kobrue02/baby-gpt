from training.classes.transformer import Transformer
from typing import Optional
from dataclasses import dataclass

import torch

@dataclass
class TrainingState:
    model: Transformer
    raw_model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: Optional[torch.cuda.amp.GradScaler | torch.amp.GradScaler] # type: ignore
    epoch: int
    iter_num: int
    lr: float
    best_val_loss: float
    config: dict
    wandb_run_id: Optional[str]
    observed_tokens_count: int
    current_loss: float = 0.0

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, model: Transformer, raw_model, optimizer, scaler) -> "TrainingState":
        # load model state
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod.") :]] = state_dict.pop(k)

        # load into the raw (uncompiled) model
        raw_model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint.get("epoch", 0)
        iter_num = checkpoint.get("iter_num", 0)
        best_val_loss = checkpoint["best_val_loss"]
        observed_tokens_count = checkpoint.get("observed_tokens_count", 0)
        wandb_run_id = checkpoint.get("wandb_run_id", None)
        config = checkpoint.get("config", {})
        lr = config.get("learning_rate", 0.0)
        current_loss = checkpoint.get("current_loss", 0.0)

        return cls(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            iter_num=iter_num,
            lr=lr,
            best_val_loss=best_val_loss,
            config=config,
            wandb_run_id=wandb_run_id,
            observed_tokens_count=observed_tokens_count,
            current_loss=current_loss,
        )

    def to_checkpoint_dict(self) -> dict:
        """Convert training state to checkpoint dictionary."""
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "wandb_run_id": self.wandb_run_id,
            "observed_tokens_count": self.observed_tokens_count,
            "current_loss": self.current_loss,
        }


@dataclass
class EvaluationState:
    epoch: int
    best_val_loss: float
    current_loss: float
    iter_num: int
    mean_perplexity: float
    train_loss: float
    val_loss: float
    lr: float