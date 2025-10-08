"""
This training script can be run on a single gpu in debug mode.

To run on a single GPU, example:
$ python -m training.pretraining
"""

import os
import numpy as np
import torch

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

    def __init__(self):
        super().__init__()
        self.device_type, self.ptdtype, self.ctx = self.setup_device()
        self.meta_vocab_size, _, _ = self.derive_vocab_size("data")
        # initialize model
        self.model = self.init_model(self.meta_vocab_size, self.config.get("compile", True))
        self.optimizer, self.scaler = self.init_optimizer_and_scaler()
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

        while ix_tuple in self.seen_batches and attempts < max_attempts:
            ix = torch.randint(
                len(data) - self.config["block_size"], (self.config["batch_size"],)
            )
            ix_tuple = tuple(ix.tolist())
            attempts += 1

        # Reset seen_batches if we've seen too many (prevent memory growth)
        if len(self.seen_batches) > 10000:
            self.seen_batches.clear()

        self.seen_batches.add(ix_tuple)
        return ix

    def get_batch(self, split: str, data_dir="data"):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(
                os.path.join(data_dir, "train_pretrain.bin"), dtype=np.uint16, mode="r"
            )
        else:
            data = np.memmap(
                os.path.join(data_dir, "val_pretrain.bin"), dtype=np.uint16, mode="r"
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
    def estimate_loss(self, data_dir="data"):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config["eval_iters"])
            for k in range(self.config["eval_iters"]):
                self.X, self.Y = self.get_batch(split, data_dir)
                with self.ctx:
                    _, loss = self.model(self.X, self.Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def forward_backward(self, data_dir="data"):
        # forward backward update, with optional gradient accumulation
        for _ in range(self.config["gradient_accumulation_steps"]):
            with self.ctx:
                _, loss = self.model(self.X, self.Y)
                loss = loss / self.config["gradient_accumulation_steps"]

            # Check for NaN/inf loss before backward pass
            if not torch.isfinite(loss):
                self.pbar.set_postfix_str(f"non-finite loss detected: {loss.item()}")
                self.pbar.set_postfix_str("skipping this batch to prevent gradient corruption")
                # Get a new batch and skip this iteration
                self.X, self.Y = self.get_batch("train", data_dir)
                continue

            self.scaler.scale(loss).backward()
            # get a new random unseen batch
            self.X, self.Y = self.get_batch("train", data_dir)

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

    def save_checkpoint(self, iter_num):
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        self.pbar.set_postfix_str(f"saving checkpoint to {self.config['out_dir']}")
        torch.save(checkpoint, os.path.join(self.config["out_dir"], "ckpt.pt"))
        self.latest_checkpoint = checkpoint

    def eval_step(self, data_dir="data", iter_num=0):
        losses = self.estimate_loss(data_dir)
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


    def training_step(self, iter_num=0, data_dir="data"):
        """Perform a single training step."""

        # determine and set the learning rate for this iteration
        self.update_optimizer_lr(iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if (
            iter_num % self.config["eval_interval"] == 0
            and self.config["master_process"]
        ):
            self.eval_step(data_dir, iter_num)

        # at the end of training, we can skip the final forward/backward pass
        if iter_num == 0 and self.config["eval_only"]:
            return

        # else, perform the forward/backward pass and clear memory
        self.forward_backward(data_dir)
        if self.device_type == "mps":
            cleanup_mps_memory()

    
    def train(self, data_dir="data"):
        """
        Train the model.
        """
        self.best_val_loss = 1e9
        self.X, self.Y = self.get_batch("train", data_dir)
        self.raw_model = self.model
        self.pbar = tqdm(total=self.config["max_iters"])
        iter_num = 0
        while iter_num < self.config["max_iters"]:
            try:
                self.pbar.update()
                self.training_step(iter_num, data_dir)
                iter_num += 1
            except KeyboardInterrupt:
                self.save_checkpoint(iter_num)
                self.pbar.close()
                print("Exiting from training early.")
                break
