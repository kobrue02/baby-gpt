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
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

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
        self.model = self.init_model(self.meta_vocab_size, self.config.get("compile", True))
        # Store raw model before compilation for checkpointing
        self.raw_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        self.optimizer, self.scaler = self.init_optimizer_and_scaler()
        # load checkpoint if resuming
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.current_loss = 0.0
        self.observed_tokens_count = 0
        self._seen_batches = set()
        if self.resume:
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

            wandb.init(
                project=self.config["wandb_project"],
                name=self.config["wandb_run_name"],
                config=self.config,
            )
            return wandb
        return None

    def find_unseen_batch(self, data: np.memmap):
        """
        Find a new unseen batch of data indices.
        """
        ix = torch.randint(
            len(data) - self.config["block_size"], (self.config["batch_size"],)
        )

        # Convert tensor to tuple for set membership check
        ix_tuple = tuple(ix.tolist())
        max_attempts = 5
        attempts = 0

        while ix_tuple in self._seen_batches and attempts < max_attempts:
            ix = torch.randint(
                len(data) - self.config["block_size"], (self.config["batch_size"],)
            )
            ix_tuple = tuple(ix.tolist())
            attempts += 1

        # Reset seen_batches if we've seen too many (prevent memory growth)
        if len(self._seen_batches) > 10000:
            self._seen_batches.clear()

        self._seen_batches.add(ix_tuple)
        return ix

    def get_batch(self, split: str):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_dir, "train_pretrain.bin"), dtype=np.uint16, mode="r"
            )
        else:
            data = np.memmap(
                os.path.join(self.data_dir, "val_pretrain.bin"), dtype=np.uint16, mode="r"
            )

        ix = self.find_unseen_batch(data)

        x = torch.stack([
            torch.from_numpy((data[i : i + self.config["block_size"]]).astype(np.int64))
            for i in ix])
        y = torch.stack([
            torch.from_numpy((data[i + 1 : i + 1 + self.config["block_size"]]).astype(np.int64))
            for i in ix])

        x, y = x.to(self.device_type), y.to(self.device_type)
        return x, y

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
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
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
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config["dtype"] == "float16"))  # type: ignore
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
            for k in range(self.config["eval_iters"]):
                self.X, self.Y = self.get_batch(split)
                with self.ctx:
                    _, loss = self.model(self.X, self.Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def forward_backward(self):
        # forward backward update, with optional gradient accumulation
        performed_backward = False
        for _ in range(self.config["gradient_accumulation_steps"]):
            with self.ctx:
                _, loss = self.model(self.X, self.Y)
                loss = loss / self.config["gradient_accumulation_steps"]
                self.observed_tokens_count += torch.numel(self.X)

            # Check for NaN/inf loss before backward pass
            if not torch.isfinite(loss):
                self.pbar.set_postfix_str(f"non-finite loss detected: {loss.item()}")
                self.pbar.set_postfix_str("skipping this batch to prevent gradient corruption")
                # Get a new batch and skip this iteration
                self.X, self.Y = self.get_batch("train")
                continue

            self.scaler.scale(loss).backward()
            performed_backward = True
            self.current_loss = loss.item() * self.config["gradient_accumulation_steps"]
            # get a new random unseen batch
            self.X, self.Y = self.get_batch("train")

        # Skip optimizer step if no backward passes were performed
        if not performed_backward:
            self.optimizer.zero_grad(set_to_none=True)
            return

        # clip the gradient
        if self.config["grad_clip"] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )

        # Check for NaN gradients after clipping
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                self.pbar.set_postfix_str(f"corrupted gradient detected in {name}")
                self.pbar.set_postfix_str("Skipping optimizer step due to NaN gradients")
                self.optimizer.zero_grad(set_to_none=True)
                return

        # step the optimizer and scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def load_checkpoint(self):
        """Load the latest checkpoint from the output directory."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            return

        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device_type)

        # Load model state
        state_dict = checkpoint["model"]
        # Remove '_orig_mod.' prefix if present (from compiled models)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Load into the raw (uncompiled) model
        self.raw_model.load_state_dict(state_dict)

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training state
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed from iteration {self.iter_num} with best val loss {self.best_val_loss:.4f}")

    def _atomic_save_checkpoint(self, checkpoint):
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    
    def save_checkpoint(self, iter_num):
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        self.pbar.set_postfix_str(f"saving checkpoint to {self.config['out_dir']}")

        # Use atomic write to prevent corruption on interruption
        self._atomic_save_checkpoint(checkpoint)

        self.latest_checkpoint = checkpoint

    def eval_step(self, iter_num=0):
        losses = self.estimate_loss()
        if self.wandb_logger:
            self.wandb_logger.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": self.lr,
                }
            )
        if losses["val"] < self.best_val_loss or self.config["always_save_checkpoint"]:
            self.best_val_loss = losses["val"]
            if iter_num > 0:
                self.save_checkpoint(iter_num)


    def training_step(self, iter_num=0):
        """Perform a single training step."""

        # determine and set the learning rate for this iteration
        self.update_optimizer_lr(iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if (
            iter_num % self.config["eval_interval"] == 0
            and self.config["master_process"]
        ):
            self.eval_step(iter_num)

        # at the end of training, we can skip the final forward/backward pass
        if iter_num == 0 and self.config["eval_only"]:
            return

        # else, perform the forward/backward pass and clear memory
        self.forward_backward()
        if self.device_type == "mps":
            cleanup_mps_memory()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()

    
    def train(self):
        """
        Train the model.
        """
        self.X, self.Y = self.get_batch("train")
        self.pbar = tqdm(total=self.config["max_iters"], initial=self.iter_num)
        while self.iter_num < self.config["max_iters"]:
            try:
                self.training_step(self.iter_num)
                self.pbar.update()
                self.pbar.set_postfix_str(f"lr {self.lr:.2e}, loss {self.current_loss:.4f}, tokens {self.observed_tokens_count:,}")
                self.iter_num += 1
            except KeyboardInterrupt:
                self.save_checkpoint(self.iter_num)
                self.pbar.close()
                print("Exiting from training early.")
                break
            except RuntimeError as e:
                self.pbar.close()
                print(f"Error during training step: {e}")
                print(str(e))
                print("Memory usage summary:")
                if self.device_type == "cuda":
                    print(torch.cuda.memory_summary())
                elif self.device_type == "mps":
                    print(get_mps_memory_info())
                print("Exiting from training.")
                break
