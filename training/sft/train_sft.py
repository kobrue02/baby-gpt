"""
This training script can be run on a single gpu in debug mode.

To run on a single GPU, example:
$ python -m training.pretraining
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from training.classes.trainer import Trainer
from training.pretraining.components.transformer import GPTWithMHA
from training.pretraining.components.blocks import GPTConfig
from training.configurator import load_sft_config
from training.util import cleanup_mps_memory


class SFTTrainer(Trainer):
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
        self.model = self.init_model()
        # Store raw model before any potential compilation
        self.raw_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        self.optimizer, self.scaler = self.init_optimizer_and_scaler()
        self._seen_batches = set()
        self.iter_num = 0
        self.best_val_loss = 1e9
        # load checkpoint if resuming
        if self.resume:
            self.load_checkpoint()
        # initialize logger
        self.wandb_logger = self.setup_logging()

    def load_and_validate_config(self):
        """
        Load and validate the training configuration.
        """
        config = load_sft_config()
        # ...
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
        # Ensure we have enough data for at least one block
        max_start_idx = len(data) - self.config["block_size"]
        if max_start_idx < 1:
            raise ValueError(
                f"Data too small: need at least {self.config['block_size'] + 1} tokens, "
                f"but got {len(data)}. Please add more training data or reduce block_size."
            )

        ix = torch.randint(
            max_start_idx, (self.config["batch_size"],)
        )

        # Convert tensor to tuple for set membership check
        ix_tuple = tuple(ix.tolist())
        max_attempts = 5
        attempts = 0

        while ix_tuple in self._seen_batches and attempts < max_attempts:
            ix = torch.randint(
                max_start_idx, (self.config["batch_size"],)
            )
            ix_tuple = tuple(ix.tolist())
            attempts += 1

        # Reset seen_batches if we've seen too many (prevent memory growth)
        if len(self._seen_batches) > 10000:
            self._seen_batches.clear()

        self._seen_batches.add(ix_tuple)
        return ix

    def get_batch(self, split: str):
        """
        Generate a batch of data for training or validation.
        Since this is SFT, we also return a mask for the loss.
        """
        tokens = np.memmap(os.path.join(self.data_dir, f"{split}_sft.bin"), dtype=np.uint16, mode="r")
        masks  = np.memmap(os.path.join(self.data_dir, f"{split}_sft_mask.bin"), dtype=np.uint8, mode="r")

        ix = self.find_unseen_batch(tokens)
        x = torch.stack([
            torch.from_numpy(tokens[i : i + self.config["block_size"]].astype(np.int64))
            for i in ix])
        y = torch.stack([
            torch.from_numpy(tokens[i + 1 : i + 1 + self.config["block_size"]].astype(np.int64))
            for i in ix])
        m = torch.stack([
            torch.from_numpy(masks[i + 1 : i + 1 + self.config["block_size"]].astype(np.float32))
            for i in ix])

        return x.to(self.device_type), y.to(self.device_type), m.to(self.device_type)


    def init_model(self) -> GPTWithMHA:
        """
        Initialize the model from a checkpoint.
        If resuming, loads from SFT checkpoint. Otherwise, loads from pretraining checkpoint.
        """
        # Determine which checkpoint to load from
        if self.resume:
            ckpt_path = os.path.join(self.config['out_dir'], 'ckpt.pt')
            print(f"Resuming SFT from {ckpt_path}")
        else:
            ckpt_path = os.path.join('out', 'ckpt.pt')
            print(f"Initializing SFT model from pretrained checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device_type)

        # Extract model configuration from checkpoint
        model_config = checkpoint['config']
        model_args = dict(
            n_layer=model_config['n_layer'],
            n_head=model_config['n_head'],
            n_embd=model_config['n_embd'],
            block_size=model_config['block_size'],
            bias=model_config['bias'],
            dropout=0.0,  # Set to 0 for fine-tuning
            vocab_size=model_config.get('vocab_size', 50304)
        )
        self.meta_vocab_size = model_config.get("vocab_size", 50304)
        print("meta_vocab_size =", self.meta_vocab_size,)
        gptconf = GPTConfig(**model_args)
        model = GPTWithMHA(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
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

    def load_checkpoint(self):
        """Load the latest SFT checkpoint from the output directory."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, starting from pretrained model")
            return

        print(f"Resuming SFT training from {checkpoint_path}")
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
        """Atomically save checkpoint to prevent corruption on interruption."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)

    def save_checkpoint(self, iter_num):
        """
        Save the latest model checkpoint.
        Args:
            iter_num (int): The current iteration number.
        """
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        print(f"saving checkpoint to {self.config['out_dir']}")

        # Use atomic write to prevent corruption on interruption
        self._atomic_save_checkpoint(checkpoint)

        self.latest_checkpoint = checkpoint

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config["eval_iters"])
            for k in range(self.config["eval_iters"]):
                self.X, self.Y, self.M = self.get_batch(split)
                with self.ctx:
                    logits, _ = self.model(self.X, self.Y)
                    loss = self._masked_loss(logits)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def _masked_loss(self, logits):
        """ Compute the SFT loss with masking the prompt tokens. """
        losses = F.cross_entropy(
            logits.view(-1, self.meta_vocab_size),
            self.Y.view(-1),
            reduction="none"
        ).view_as(self.Y)
        return (losses * self.M).sum() / self.M.sum()
    
    def _accumulate_gradients(self) -> bool:
        """ Accumulate gradients over multiple mini-batches. """
        performed_backward = False
        for _ in range(self.config["gradient_accumulation_steps"]):
            with self.ctx:
                logits, _ = self.model(self.X, self.Y)
                loss = self._masked_loss(logits)
                loss = loss / self.config["gradient_accumulation_steps"]

            # Check for NaN/inf loss before backward pass
            if not torch.isfinite(loss):
                self.X, self.Y, self.M = self.get_batch("train")
                continue

            self.scaler.scale(loss).backward()
            performed_backward = True
        return performed_backward
    
    def _clip_gradients(self):
        """ Clip the gradients to prevent explosion. """
        if self.config["grad_clip"] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )

    def _validate_gradients(self) -> bool:
        """ check for NaN gradients after clipping """
        for _, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                self.optimizer.zero_grad(set_to_none=True)
                return False
        return True
    
    def _step(self):
        """ Step the optimizer and scaler. """
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def forward_backward(self):
        # forward backward update, with gradient accumulation
        performed_backward = self._accumulate_gradients()

        # skip optimizer step if no backward passes were performed
        if not performed_backward:
            self.optimizer.zero_grad(set_to_none=True)
            return

        # clip the gradient
        self._clip_gradients()

        # if any gradients are NaN/inf, skip this update
        if not self._validate_gradients():
            self.optimizer.zero_grad(set_to_none=True)
            return

        # step the optimizer and scaler
        self._step()

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
        """ Perform a single training step. """

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

        # get a new random unseen batch
        self.X, self.Y, self.M = self.get_batch("train")

    
    def train(self):
        """
        Train the model.
        """
        self.X, self.Y, self.M = self.get_batch("train")
        self.pbar = tqdm(total=self.config["max_iters"], initial=self.iter_num)
        while self.iter_num < self.config["max_iters"]:
            try:
                self.training_step(self.iter_num)
                self.pbar.update()
                self.iter_num += 1
            except KeyboardInterrupt:
                self.save_checkpoint(self.iter_num)
                self.pbar.close()
                print("Exiting from training early.")
                break
