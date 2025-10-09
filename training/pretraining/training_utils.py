"""
This training script can be run on a single gpu in debug mode.

To run on a single GPU, example:
$ python -m training.pretraining
"""

import os
import numpy as np
import torch
import warnings

# Suppress Pydantic warnings about Field attributes
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)

from tqdm import tqdm
from training.classes.trainer import Trainer
from training.pretraining.components.transformer import GPTWithMHA
from training.pretraining.components.blocks import GPTConfig
from training.configurator import load_config
from training.util import cleanup_mps_memory, get_mps_memory_info


class PreTrainer(Trainer):
    """
    Trainer class to handle model training.
    """

    def __init__(self, resume=False, data_dir="data"):
        super().__init__()
        self.resume = resume
        self.data_dir = data_dir

        self.device_type, self.ptdtype, self.ctx = self.setup_device()
        self.meta_vocab_size, _, _ = self.derive_vocab_size(self.data_dir)
        
        # initialize model
        self.model = self.init_model(
            self.meta_vocab_size, self.config.get("compile", True)
        )
       
        # store raw model before compilation for checkpointing
        self.raw_model = (
            self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        )
        self.optimizer, self.scaler = self.init_optimizer_and_scaler()
        
        # initialize training state, will be overridden if resuming from checkpoint
        self.epoch = 0
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.current_loss = 0.0
        self.observed_tokens_count = 0
        self.wandb_run_id = None
        
        # training data setup
        self.train_data_len = self._get_dataset_length("train")
        self.steps_per_epoch = self.train_data_len // (
            self.config["batch_size"] * self.config["block_size"]
        )
        self.total_steps = self.steps_per_epoch * self.config["n_epochs"]

        if self.resume:  # load existing checkpoint
            self.load_checkpoint()

        # initialize logger
        self.wandb_logger = self.setup_logging()

    def load_and_validate_config(self):
        """
        Load and validate the training configuration.
        """
        config = load_config()

        # validate config values
        assert config["device"] in [
            "cpu",
            "cuda",
            "mps",
        ], "Only cpu, cuda, and mps devices are supported in this script."
        assert (
            config["n_embd"] % config["n_head"] == 0
        ), "Embedding size must be divisible by number of heads."

        print(f"tokens per iteration will be: {config['tokens_per_iter']:,}")
        self.config = config

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
                self.wandb_run_id = wandb.run.id
            return wandb
        return None

    def _get_dataset_length(self, split: str):
        """Get the length of the dataset in tokens."""
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_dir, "train_pretrain.bin"),
                dtype=np.uint16,
                mode="r",
            )
        else:
            data = np.memmap(
                os.path.join(self.data_dir, "val_pretrain.bin"),
                dtype=np.uint16,
                mode="r",
            )
        return len(data)

    def _create_dataloader_indices(self, data_len: int):
        """Create shuffled indices for epoch-based training."""
        # number of sequences we can extract
        num_sequences = (data_len - self.config["block_size"]) // self.config["block_size"]

        # sequential starting indices
        indices = torch.arange(0, num_sequences * self.config["block_size"], self.config["block_size"])

        # shuffle indices for a given epoch
        return indices[torch.randperm(len(indices))]


    def get_batch(self, split: str, indices: torch.Tensor, batch_idx: int):
        """
        Get a batch of data using pre-shuffled indices.

        Args:
            split: 'train' or 'val'
            indices: Shuffled indices for this epoch
            batch_idx: Current batch index within the epoch
        """
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_dir, "train_pretrain.bin"),
                dtype=np.uint16,
                mode="r",
            )
        else:
            data = np.memmap(
                os.path.join(self.data_dir, "val_pretrain.bin"),
                dtype=np.uint16,
                mode="r",
            )

        # get batch_size indices from the shuffled index array
        start_idx = batch_idx * self.config["batch_size"]
        end_idx = min(start_idx + self.config["batch_size"], len(indices))
        batch_indices = indices[start_idx:end_idx]

        # load sequences at these positions
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + self.config["block_size"]]).astype(np.int64)
                )
                for i in batch_indices
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.config["block_size"]]).astype(np.int64)
                )
                for i in batch_indices
            ]
        )

        return x.to(self.device_type), y.to(self.device_type)

    def init_model(self, meta_vocab_size, compile_model=True) -> GPTWithMHA:
        """
        Initialize the model from scratch.
        """
        # model init
        model_args = dict(
            n_layer=self.config["n_layer"],
            n_head=self.config["n_head"],
            n_embd=self.config["n_embd"],
            block_size=self.config["block_size"],
            bias=self.config["bias"],
            dropout=self.config["dropout"],
        )

        # determine the vocab size we'll use for training
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )

        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        model_args["device"] = self.config["device"]
        gptconf = GPTConfig(**model_args)
        model = GPTWithMHA(gptconf)

        if compile_model:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0

        return model.to(self.device_type)

    def init_optimizer_and_scaler(self):
        """Initialize the optimizer and gradient scaler."""
        if torch.__version__ <= "2.4":
            scaler = torch.cuda.amp.GradScaler(enabled=(self.config["dtype"] == "float16"))  # type: ignore
        else:
            scaler = torch.amp.GradScaler(enabled=(self.config["dtype"] == "float16"))  # type: ignore
        optimizer = self.model.configure_optimizers(
            self.config["weight_decay"],
            self.config["learning_rate"],
            (self.config["beta1"], self.config["beta2"]),
            self.device_type,
        )
        return optimizer, scaler

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config["eval_iters"])
            # Create temporary indices for evaluation
            data_len = self._get_dataset_length(split)
            eval_indices = self._create_dataloader_indices(data_len)
            for k in range(self.config["eval_iters"]):
                if k >= len(eval_indices) // self.config["batch_size"]:
                    break  # Don't go past the dataset
                self.X, self.Y = self.get_batch(split, eval_indices, k)
                with self.ctx:
                    _, loss = self.model(self.X, self.Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def _perform_gradient_accumulation_steps(self, train_indices, batch_idx):
        """
        Perform gradient accumulation over multiple micro-steps.
        Returns True if at least one backward pass was performed, False otherwise.
        """
        performed_backward = False
        for micro_step in range(self.config["gradient_accumulation_steps"]):
            # get next batch for gradient accumulation
            current_batch_idx = (
                batch_idx * self.config["gradient_accumulation_steps"] + micro_step
            )
            if current_batch_idx >= len(train_indices) // self.config["batch_size"]:
                break  # don't go past the epoch

            self.X, self.Y = self.get_batch("train", train_indices, current_batch_idx)

            with self.ctx:
                _, loss = self.model(self.X, self.Y)
                loss = loss / self.config["gradient_accumulation_steps"]
                self.observed_tokens_count += torch.numel(self.X)

            # check for NaN/inf loss before backward pass
            if not torch.isfinite(loss):
                self.pbar.set_postfix_str(f"non-finite loss detected: {loss.item()}")
                self.pbar.set_postfix_str(
                    "skipping this batch to prevent gradient corruption"
                )
                continue

            self.scaler.scale(loss).backward()
            performed_backward = True
            self.current_loss = loss.item() * self.config["gradient_accumulation_steps"]
       
        return performed_backward
    
    def _validate_gradients_clipped(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                self.pbar.set_postfix_str(f"corrupted gradient detected in {name}")
                self.pbar.set_postfix_str(
                    "Skipping optimizer step due to NaN gradients"
                )
                self.optimizer.zero_grad(set_to_none=True)
                return False
        return True
    
    def _step(self):
        """ Step the optimizer and scaler after gradient accumulation. """
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def forward_backward(self, train_indices, batch_idx):
        """ Perform the forward and backward pass, with gradient accumulation if needed. """
        # forward backward update, with gradient accumulation
        performed_backward = self._perform_gradient_accumulation_steps(train_indices, batch_idx)
        # skip optimizer step if no backward passes were performed
        if not performed_backward:
            self.optimizer.zero_grad(set_to_none=True)
            return
        # clip the gradient
        if self.config["grad_clip"] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )
        # check for NaN gradients after clipping
        if not self._validate_gradients_clipped():
            return
        else: # step the optimizer and scaler
            self._step()

    def _validate_checkpoint(self, checkpoint_path):
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            return
        print(f"Resuming training from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=self.device_type)

    def _load_states(self, checkpoint):
        # load model state
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod.") :]] = state_dict.pop(k)

        # load into the raw (uncompiled) model
        self.raw_model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epoch = checkpoint.get("epoch", 0)
        self.iter_num = checkpoint.get("iter_num", 0)
        self.best_val_loss = checkpoint["best_val_loss"]
        self.observed_tokens_count = checkpoint.get("observed_tokens_count", 0)
        self.wandb_run_id = checkpoint.get("wandb_run_id", None)

        wandb_msg = f" (wandb run ID: {self.wandb_run_id})" if self.wandb_run_id else ""
        print(
            f"Resumed from epoch {self.epoch}, iteration {self.iter_num} with best val loss {self.best_val_loss:.4f}{wandb_msg}"
        )

    def load_checkpoint(self):
        """Load the latest checkpoint from the output directory."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        checkpoint = self._validate_checkpoint(checkpoint_path)
        if checkpoint is None:
            return  # No checkpoint to load
        else:
            self._load_states(checkpoint)

    def _atomic_save_checkpoint(self, checkpoint):
        """ Atomically save the checkpoint to prevent corruption on interruption. """
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

    def save_checkpoint(self, epoch, iter_num):
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
            "observed_tokens_count": self.observed_tokens_count,
            "wandb_run_id": self.wandb_run_id,
            "config": self.config,
        }
        self.pbar.set_postfix_str(f"saving checkpoint to {self.config['out_dir']}")

        # Use atomic write to prevent corruption on interruption
        self._atomic_save_checkpoint(checkpoint)

        self.latest_checkpoint = checkpoint

    def eval_step(self, epoch, iter_num=0):
        losses = self.estimate_loss()
        if self.wandb_logger:
            self.wandb_logger.log(
                {
                    "epoch": epoch,
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": self.lr,
                }
            )
        if losses["val"] < self.best_val_loss or self.config["always_save_checkpoint"]:
            self.best_val_loss = losses["val"]
            if iter_num > 0:
                self.save_checkpoint(epoch, iter_num)

    def training_step(self, epoch, iter_num, train_indices, batch_idx):
        """Perform a single training step."""

        # determine and set the learning rate for this iteration
        self.update_optimizer_lr(iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if (
            iter_num % self.config["eval_interval"] == 0
            and self.config["master_process"]
        ):
            self.eval_step(epoch, iter_num)

        # at the end of training, we can skip the final forward/backward pass
        if iter_num == 0 and self.config["eval_only"]:
            return

        # else, perform the forward/backward pass and clear memory
        self.forward_backward(train_indices, batch_idx)
        
        if self.device_type == "mps":
            cleanup_mps_memory()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()

    def _runtime_error_exit(self, e: RuntimeError):
        """Handle RuntimeError during training by saving checkpoint and exiting gracefully."""
        self.pbar.close()

        print(f"Error during training step: {e}")
        print(str(e))
        print("Memory usage summary:")

        if self.device_type == "cuda":
            print(torch.cuda.memory_summary())
        elif self.device_type == "mps":
            print(get_mps_memory_info())

        print("Exiting from training.")

    def train(self):
        """
        Train the model using epoch-based training.
        """
        for epoch in range(self.epoch, self.config["n_epochs"]):
            train_indices = self._create_dataloader_indices(self.train_data_len)
            batches_per_epoch = (
                len(train_indices) // self.config["batch_size"]
            ) // self.config["gradient_accumulation_steps"]
            self.pbar = tqdm(
                total=batches_per_epoch,
                initial=self.iter_num % batches_per_epoch,
                desc=f"epoch {epoch+1}/{self.config['n_epochs']}",
            )
            try:
                for batch_idx in range(batches_per_epoch):
                    self.training_step(epoch, self.iter_num, train_indices, batch_idx)
                    self.pbar.update()
                    self.pbar.set_postfix_str(
                        f"lr {self.lr:.2e}, loss {self.current_loss:.4f}, "
                        f"tokens {self.observed_tokens_count:,}"
                    )
                    self.iter_num += 1

                # save checkpoint at end of epoch
                if self.config["master_process"]:
                    self.save_checkpoint(epoch + 1, self.iter_num)

            except KeyboardInterrupt:
                self.save_checkpoint(epoch, self.iter_num)
                self.pbar.close()
                print("Exiting from training early.")
                break

            except RuntimeError as e:
                self._runtime_error_exit(e)
                break

        self.pbar.close()
        print(f"exiting training - epoch {epoch}, iter {self.iter_num}")
