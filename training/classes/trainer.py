import torch
import os
import math
import pickle
import tiktoken

from abc import ABC
from typing import Tuple, Dict, Any
from tqdm import tqdm

from training.classes.states import TrainingState, EvaluationState


class Trainer(ABC):
    """
    Abstract Trainer class that serves as a base for specific training implementations.
    """

    def __init__(self):
        self.config: Dict[str, Any]
        self.load_and_validate_config()
        self.device_type: str
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.training_state: TrainingState
        self.eval_state: EvaluationState
        self.pbar: tqdm  # Progress bar for training

    # Properties for backward compatibility and cleaner access
    @property
    def model(self):
        return self.training_state.model

    @property
    def raw_model(self):
        return self.training_state.raw_model

    @property
    def optimizer(self):
        return self.training_state.optimizer

    @property
    def scaler(self):
        return self.training_state.scaler

    @property
    def scheduler(self):
        return self.training_state.scheduler

    @property
    def epoch(self):
        return self.training_state.epoch

    @property
    def iter_num(self):
        return self.training_state.iter_num

    @iter_num.setter
    def iter_num(self, value):
        self.training_state.iter_num = value

    @property
    def lr(self):
        return self.training_state.lr

    @lr.setter
    def lr(self, value):
        self.training_state.lr = value

    @property
    def best_val_loss(self):
        return self.training_state.best_val_loss

    @property
    def wandb_run_id(self):
        return self.training_state.wandb_run_id

    @wandb_run_id.setter
    def wandb_run_id(self, value):
        self.training_state.wandb_run_id = value

    @property
    def observed_tokens_count(self):
        return self.training_state.observed_tokens_count

    @observed_tokens_count.setter
    def observed_tokens_count(self, value):
        self.training_state.observed_tokens_count = value

    @property
    def predicted_tokens_count(self):
        return self.training_state.predicted_tokens_count

    @predicted_tokens_count.setter
    def predicted_tokens_count(self, value):
        self.training_state.predicted_tokens_count = value

    @property
    def current_loss(self):
        return self.training_state.current_loss

    @current_loss.setter
    def current_loss(self, value):
        self.training_state.current_loss = value

    def load_and_validate_config(self):
        """
        Load and validate the training configuration.
        """
        pass

    def setup_device(self) -> Tuple[str, torch.dtype, torch.autocast]:
        """
        Setup the device and context to use for training.
        Returns:
            device_type (str): The type of device ('cpu', 'cuda', 'mps').
            ptdtype (torch.dtype): The PyTorch data type to use.
            ctx (torch.autocast): The context manager for automatic mixed precision.
        """
        if self.config["master_process"]:
            os.makedirs(self.config["out_dir"], exist_ok=True)
        torch.manual_seed(1337 + self.config["seed_offset"])
        # torch.backends.cuda.matmul.fp32_precision = 'tf32' # use tf32 for matmul
        # torch.backends.cudnn.conv.fp32_precision = 'tf32' # type: ignore
        device_type = self.config["device"]
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config["dtype"]]
        if (
            torch.__version__ >= "2.4"
        ):  # load autocast from the correct place depending on torch version
            ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # type: ignore
        else:
            ctx = torch.cuda.amp.autocast(dtype=ptdtype)  # type: ignore
        return device_type, ptdtype, ctx

    def get_batch(
        self, split: str, data_dir="data"
    ) -> (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Generate a batch of data of inputs x and targets y with np.memmap
        Args:
            split (str): The data split to use ('train' or 'val').
            data_dir (str): The directory where the data is stored.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor x and target tensor y.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def derive_vocab_size(self, data_dir):
        """
        Attempt to derive the vocab size from the dataset's meta.pkl file.
        """

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, "meta.pkl")
        meta_vocab_size = None
        encode, decode = None, None

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

            # TODO want to make this more general to arbitrary encoder/decoder schemes
            stoi, itos = meta["stoi"], meta["itos"]
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: "".join([itos[i] for i in l])
        else:
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)

        return meta_vocab_size, encode, decode

    def init_model(self, *args, **kwargs):
        """
        Initialize the model from scratch.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, *args, **kwargs) -> Dict[str, float]:
        """
        Estimate the loss over either split using many batches, so that we get a more accurate estimate
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it) -> float:
        """
        Get the learning rate for the current iteration using a cosine decay schedule with warmup.
        Args:
            it (int): The current iteration number.
        Returns:
            float: The learning rate for the current iteration.
        """
        # 1) linear warmup for warmup_iters steps
        if it < self.config["warmup_iters"]:
            return (
                self.config["learning_rate"]
                * (it + 1)
                / (self.config["warmup_iters"] + 1)
            )
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config["lr_decay_iters"]:
            return self.config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config["warmup_iters"]) / (
            self.config["lr_decay_iters"] - self.config["warmup_iters"]
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config["min_lr"] + coeff * (
            self.config["learning_rate"] - self.config["min_lr"]
        )

    def update_optimizer_lr(self, iter_num):
        """
        Get and set the learning rate for the current iteration.
        Uses PyTorch scheduler if available, otherwise falls back to manual schedule.
        Args:
            iter_num (int): The current iteration number.
        """
        if self.training_state.scheduler is not None:
            # Use PyTorch scheduler
            self.training_state.lr = self.training_state.optimizer.param_groups[0]["lr"]
        else:
            # Fallback to manual scheduling
            self.training_state.lr = (
                self.get_lr(iter_num)
                if self.config["decay_lr"]
                else self.config["learning_rate"]
            )
            for param_group in self.training_state.optimizer.param_groups:
                param_group["lr"] = self.training_state.lr

    def init_optimizer_and_scaler(self, model=None):
        """Initialize the optimizer and gradient scaler."""
        if torch.__version__ <= "2.4":
            scaler = torch.cuda.amp.GradScaler(enabled=(self.config["dtype"] == "float16"))  # type: ignore
        else:
            scaler = torch.amp.GradScaler(enabled=(self.config["dtype"] == "float16"))  # type: ignore
        # Use provided model or fall back to self.model property
        model_to_use = model if model is not None else self.model
        optimizer = model_to_use.configure_optimizers(
            self.config["weight_decay"],
            self.config["learning_rate"],
            (self.config["beta1"], self.config["beta2"]),
            self.device_type,
        )
        return optimizer, scaler

    def init_scheduler(self, optimizer, total_steps):
        """
        Initialize a PyTorch LR scheduler appropriate for pretraining.
        Uses CosineAnnealingLR with warmup for stable training.

        Args:
            optimizer: The optimizer to schedule
            total_steps: Total number of training steps

        Returns:
            torch.optim.lr_scheduler or None if scheduler is disabled
        """
        if not self.config.get("use_pytorch_scheduler", False):
            return None

        warmup_iters = self.config.get("warmup_iters", 0)

        if warmup_iters > 0:
            # Create a warmup + cosine annealing scheduler
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of base LR
                end_factor=1.0,
                total_iters=warmup_iters
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_iters,
                eta_min=self.config.get("min_lr", 0.0)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_iters]
            )
        else:
            # Just use cosine annealing without warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.config.get("min_lr", 0.0)
            )

        return scheduler

    def init_eval_state(self):
        """Initialize the evaluation state with default values."""
        return EvaluationState(
            epoch=0,
            best_val_loss=float("inf"),
            current_loss=0.0,
            iter_num=0,
            mean_perplexity=float("inf"),
            coherence_rate=0.0,
            token_entropy=0.0,
            train_loss=0.0,
            val_loss=0.0,
            lr=0.0,
        )

    def setup_logging(self):
        """Setup logging with Weights & Biases if enabled in the config."""
        if self.config["wandb_log"] and self.config["master_process"]:
            import wandb

            # Resume existing run if we have a run ID, otherwise start new run
            if self.wandb_run_id:
                wandb.init(
                    project=self.config["wandb_project"],
                    id=self.wandb_run_id,
                    resume="must",
                    config=self.config,
                )
            else:
                wandb.init(
                    project=self.config["wandb_project"],
                    name=self.config["wandb_run_name"],
                    config=self.config,
                )
                # Store the run ID for future checkpoints
                self.wandb_run_id = wandb.run.id # type: ignore
            return wandb
        return None

    def _validate_gradients_clipped(self):
        """Validate that gradients are finite after clipping."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                if hasattr(self, 'pbar'):
                    self.pbar.set_postfix_str(f"corrupted gradient detected in {name}")
                    self.pbar.set_postfix_str(
                        "Skipping optimizer step due to NaN gradients"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                return False
        return True

    def _step(self):
        """Step the optimizer and scaler after gradient accumulation."""
        assert self.scaler is not None, "Scaler should be initialized"
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # step the scheduler if using PyTorch scheduler
        if self.training_state.scheduler is not None:
            self.training_state.scheduler.step()

    def _validate_checkpoint(self, checkpoint_path):
        """Validate and load a checkpoint file."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            return
        print(f"Resuming training from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=self.device_type)

    def _load_states(self, checkpoint):
        """Load training and evaluation states from checkpoint."""
        self.training_state = TrainingState.from_checkpoint(
            checkpoint, self.model, self.raw_model, self.optimizer, self.scaler, self.scheduler
        )
        wandb_msg = f" (wandb run ID: {self.training_state.wandb_run_id})" if self.training_state.wandb_run_id else ""
        print(
            f"Resumed from epoch {self.training_state.epoch}, \
            iteration {self.training_state.iter_num} \
            with best val loss {self.training_state.best_val_loss:.4f}{wandb_msg}"
        )

    def load_checkpoint(self):
        """Load the latest checkpoint from the output directory."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        checkpoint = self._validate_checkpoint(checkpoint_path)
        if checkpoint is None:
            return  # No checkpoint to load
        else:
            self._load_states(checkpoint)

    def _atomic_save_checkpoint(self, checkpoint_dict: dict):
        """Atomically save the checkpoint to prevent corruption on interruption."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint_dict, temp_path)
        os.replace(temp_path, checkpoint_path)

    def save_checkpoint(self):
        """Save the current training state to a checkpoint file."""
        if hasattr(self, 'pbar'):
            self.pbar.set_postfix_str(f"saving checkpoint to {self.config['out_dir']}")
        # Use atomic write to prevent corruption on interruption
        self._atomic_save_checkpoint(self.training_state.to_checkpoint_dict())

    def forward_backward(self, *args, **kwargs):
        """
        Perform the forward and backward pass, with gradient accumulation if needed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def eval_step(self, data_dir="data", iter_num=0):
        """Evaluate the model and log results. Save a checkpoint if the model is the best seen so far."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def training_step(self, *args, **kwargs):
        """Perform a single training step."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def train(self, data_dir="data"):
        """
        Train the model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
