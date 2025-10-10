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
from training.classes.states import TrainingState
from training.pretraining.components.transformer import GPTWithMHA
from training.pretraining.components.blocks import GPTConfig
from training.configurator import load_config
from training.util import cleanup_mps_memory, get_mps_memory_info
from training.pretraining.metrics.periodic_evals import PeriodicEval


class PreTrainer(Trainer):
    """
    Trainer class to handle model training.
    """

    def __init__(self, resume=False, data_dir="data"):
        super().__init__()
        self.resume = resume
        self.data_dir = data_dir

        self.device_type, self.ptdtype, self.ctx = self.setup_device()
        self.meta_vocab_size, self.encode, self.decode = self.derive_vocab_size(self.data_dir)

        # init training state, will be overwritten if resuming from checkpoint
        self.training_state = self.init_training_state()
        self.eval_state = self.init_eval_state()

        # training data setup
        self.train_data_len = self._get_dataset_length("train")
        self.steps_per_epoch = self.train_data_len // (
            self.config["batch_size"] * self.config["block_size"]
        )
        self.total_steps = self.steps_per_epoch * self.config["n_epochs"]

        if self.resume:  # load existing checkpoint
            self.load_checkpoint()

        # initialize logger and evaluator
        self.wandb_logger = self.setup_logging()
        self.evaluator = PeriodicEval('gpt2')

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

    def init_training_state(self):
        """
        Initialize the training state.
        Returns:
            TrainingState: The initialized training state.
        """
        # initialize model
        model = self.init_model(
            self.meta_vocab_size, self.config.get("compile", True)
        )
        # store raw model before compilation for checkpointing
        raw_model: torch.nn.Module = (
            model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore
        )
        # initialize optimizer and scaler
        optimizer, scaler = self.init_optimizer_and_scaler()
        lr = self.config["learning_rate"]
        return TrainingState(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=0,
            lr=lr,
            iter_num=0,
            best_val_loss=float("inf"),
            config=self.config,
            wandb_run_id=None,  # Will be set by setup_logging if needed
            observed_tokens_count=0,
        )

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

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
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

            assert self.scaler is not None, "Scaler should be initialized"
            self.scaler.scale(loss).backward()
            performed_backward = True
            self.current_loss = loss.item() * self.config["gradient_accumulation_steps"]

        return performed_backward

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
            assert self.scaler is not None, "Scaler should be initialized"
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )
        # check for NaN gradients after clipping
        if not self._validate_gradients_clipped():
            return
        else: # step the optimizer and scaler
            self._step()

    def get_mean_perplexity(self):
        # list of encodings
        generations_batch = self._generate_random_completions(
            max_new_tokens=50, num_samples=5, temperature=0.8
        )
        mean_pplx = self.evaluator.perplexity(generations_batch)
        return mean_pplx

    def eval_step(self, epoch, iter_num=0):
        """Evaluate the model and log results. Save a checkpoint if the model is the best seen so far."""
        self.model.eval()
        losses = self.estimate_loss()
        mean_pplx = self.get_mean_perplexity()

        # Update evaluation state
        self.eval_state.epoch = epoch
        self.eval_state.iter_num = iter_num
        self.eval_state.train_loss = losses["train"]
        self.eval_state.val_loss = losses["val"]
        self.eval_state.mean_perplexity = mean_pplx
        self.eval_state.lr = self.lr
        self.eval_state.current_loss = self.current_loss

        if self.wandb_logger:
            self.wandb_logger.log(
                {
                    "epoch": epoch,
                    "iter": iter_num,
                    "mean_perplexity": mean_pplx,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": self.lr,
                }
            )
        if losses["val"] < self.training_state.best_val_loss or self.config["always_save_checkpoint"]:
            self.training_state.best_val_loss = losses["val"]
            self.eval_state.best_val_loss = losses["val"]
            if iter_num > 0:
                self.save_checkpoint()

        self.training_state.model.train()

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

    def _generate_random_completions(self, max_new_tokens=200, num_samples=10, temperature=0.8, top_k=200) -> list:
        """ Generate random completions from the model for qualitative evaluation. """
        prompts = ["the", "once upon a time", "the meaning of life is", "the book is", "the capital of france is"]
        completions = []
        contexts = [
            torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device_type)[None, ...]
            for prompt in prompts[:num_samples]
            ]
        for context in contexts:
            # run generation
            with torch.no_grad():
                with self.ctx:
                    for _ in range(num_samples):
                        y = self.training_state.model.generate(
                            context, max_new_tokens, temperature=temperature, top_k=top_k
                        )
                        completions.append(y)
        return completions

    def train(self):
        """
        Train the model using epoch-based training.
        """
        for epoch in range(self.training_state.epoch, self.config["n_epochs"]):
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
                    self.training_step(epoch, self.training_state.iter_num, train_indices, batch_idx)
                    self.pbar.update()
                    self.pbar.set_postfix_str(
                        f"lr {self.training_state.lr:.2e}, loss {self.current_loss:.4f}, "
                        f"tokens {self.training_state.observed_tokens_count:,}"
                    )
                    self.iter_num += 1

                # save checkpoint at end of epoch
                if self.config["master_process"]:
                    self.training_state.epoch = epoch + 1
                    self.save_checkpoint()

            except KeyboardInterrupt:
                self.save_checkpoint()
                self.pbar.close()
                print("Exiting from training early.")
                break

            except RuntimeError as e:
                self._runtime_error_exit(e)
                break

        self.pbar.close()
        print(f"exiting training - epoch {epoch}, iter {self.iter_num}")
