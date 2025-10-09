import torch
import os
import math
import pickle

from abc import ABC
from typing import Tuple, Dict, Any
from tqdm import tqdm


class Trainer(ABC):
    """
    Abstract Trainer class that serves as a base for specific training implementations.
    """

    def __init__(self):
        self.config: Dict[str, Any]
        self.load_and_validate_config()

        self.device_type: str
        self.latest_checkpoint: dict
        self.best_val_loss: float
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.lr: float
        self.optimizer: torch.optim.Optimizer
        self.scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler  # type: ignore
        self.model: Any
        self.raw_model: Any

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
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        return meta_vocab_size, iter_num, best_val_loss

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
        Get and set the learning rate for the current iteration
        Args:
            iter_num (int): The current iteration number.
        """
        self.lr = (
            self.get_lr(iter_num)
            if self.config["decay_lr"]
            else self.config["learning_rate"]
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def forward_backward(self, *args, **kwargs):
        """
        Perform the forward and backward pass, with gradient accumulation if needed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

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
        torch.save(checkpoint, os.path.join(self.config["out_dir"], "ckpt.pt"))
        self.latest_checkpoint = checkpoint

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
