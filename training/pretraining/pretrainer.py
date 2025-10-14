"""
This training script can be run on a single gpu in debug mode.

To run on a single GPU, example:
$ python -m training.pretraining
"""

import gc
import os
import numpy as np
import torch
import time
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._generate_schema"
)

from tqdm import tqdm
from typing import Optional
from training.classes.trainer import Trainer
from training.classes.states import TrainingState, EvaluationState
from training.classes.transformer import Transformer
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

        # training data setup
        self.train_data_len = self._get_dataset_length("train")
        self.steps_per_epoch = self.train_data_len // (
            self.config["batch_size"] * self.config["block_size"]
        )
        self.total_steps = self.steps_per_epoch * self.config["n_epochs"]

        # calculate batches per epoch to validate intervals
        self._calculate_batches_per_epoch()
        self._validate_and_adjust_intervals()

        # init training state, will be overwritten if resuming from checkpoint
        self.training_state = self.init_training_state()
        self.eval_state = self.init_eval_state()

        self._train_indices = torch.Tensor([])
        self._eval_indices = torch.Tensor([])
        self._batch_idx = 0

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
        # initialize optimizer and scaler, passing model explicitly
        optimizer, scaler = self.init_optimizer_and_scaler(model=model)

        # initialize scheduler with total steps
        scheduler = self.init_scheduler(optimizer, self.total_steps)

        lr = self.config["learning_rate"]
        return TrainingState(
            model=model,
            raw_model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=0,
            batch_process_time=float("nan"),
            lr=lr,
            iter_num=0,
            stage_iter_num=0,
            best_val_loss=float("inf"),
            config=self.config,
            wandb_run_id=None,  # Will be set by setup_logging if needed
            observed_tokens_count=0,
            predicted_tokens_count=0,
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

    def _calculate_batches_per_epoch(self):
        """Calculate batches per epoch from training data."""
        num_sequences = (self.train_data_len - self.config["block_size"]) // self.config["block_size"]
        self._batches_per_epoch = (
            num_sequences // self.config["batch_size"]
        ) // self.config["gradient_accumulation_steps"]

    def _validate_and_adjust_intervals(self):
        """Validate and adjust eval/log intervals for small datasets."""
        if self._batches_per_epoch <= 0:
            raise ValueError(
                f"Dataset too small: only {self._batches_per_epoch} batches per epoch. "
                f"Reduce batch_size, block_size, or gradient_accumulation_steps."
            )

        # Adjust eval_interval if it's larger than batches per epoch
        if self._batches_per_epoch < self.config["eval_interval"]:
            old_eval = self.config["eval_interval"]
            # Set to half of batches per epoch, but at least 1
            self.config["eval_interval"] = max(1, self._batches_per_epoch // 2)
            print(
                f"Warning: eval_interval ({old_eval}) > batches_per_epoch ({self._batches_per_epoch}). "
                f"Adjusting eval_interval to {self.config['eval_interval']}"
            )

        # Adjust log_interval if it's larger than batches per epoch
        if self._batches_per_epoch < self.config["log_interval"]:
            old_log = self.config["log_interval"]
            # Set to quarter of batches per epoch, but at least 1
            self.config["log_interval"] = max(1, self._batches_per_epoch // 4)
            print(
                f"Warning: log_interval ({old_log}) > batches_per_epoch ({self._batches_per_epoch}). "
                f"Adjusting log_interval to {self.config['log_interval']}"
            )

    def _create_dataloader_indices(self, data_len: Optional[int] = None, split: str = "train"):
        """Create shuffled indices for epoch-based training."""
        if data_len is None or split == "train":
            data_len = self.train_data_len
        else:
            data_len = self._get_dataset_length(split)

        # number of sequences we can extract
        num_sequences = (data_len - self.config["block_size"]) // self.config["block_size"]

        # handle case where dataset is too small
        if num_sequences <= 0:
            raise ValueError(
                f"Dataset ({split}) too small: {data_len} tokens, but need at least "
                f"{2 * self.config['block_size']} tokens (2 * block_size) to create sequences."
            )

        # sequential starting indices
        indices = torch.arange(0, num_sequences * self.config["block_size"], self.config["block_size"])

        # shuffle indices for a given epoch
        indices = indices[torch.randperm(len(indices))]

        if split == "val":
            self._eval_indices = indices
            return

        else:
            self._train_indices = indices
            # Note: _batches_per_epoch already calculated in __init__


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
            self._create_dataloader_indices(data_len=data_len, split=split)
            # Use the correct indices based on split
            indices = self._eval_indices if split == "val" else self._train_indices
            for k in range(self.config["eval_iters"]):
                if k >= len(indices) // self.config["batch_size"]:
                    break  # Don't go past the dataset
                self.X, self.Y = self.get_batch(split, indices, k)
                with self.ctx:
                    _, loss = self.model(self.X, self.Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        return out
    
    def _perform_gradient_accumulation_steps(self):
        """
        Perform gradient accumulation over multiple micro-steps.

        Args:
            train_indices: Shuffled indices for the current epoch
            base_batch_idx: The base batch index (will process gradient_accumulation_steps batches starting from this)

        Returns:
            True if at least one backward pass was performed, False otherwise.
        """
        performed_backward = False
        accumulated_loss = 0.0
        successful_steps = 0

        num_accumulation_steps = self.config["gradient_accumulation_steps"]
        max_batches_available = len(self._train_indices) // self.config["batch_size"]

        for micro_step in range(num_accumulation_steps):
            # each base_batch_idx covers num_accumulation_steps micro-batches
            micro_batch_idx = self._batch_idx * num_accumulation_steps + micro_step
            if micro_batch_idx >= max_batches_available:
                break # stop if we've exhausted the dataset

            self.X, self.Y = self.get_batch("train", self._train_indices, micro_batch_idx)

            with self.ctx:
                _, loss = self.model(self.X, self.Y)
                # scale loss (to account for accumulation)
                loss = loss / num_accumulation_steps
                # track token counts for stats
                self.training_state.observed_tokens_count += torch.numel(self.X)
                self.training_state.predicted_tokens_count += torch.numel(self.Y)

            if not torch.isfinite(loss):
                self.pbar.set_postfix_str(
                    f"non-finite loss: {loss.item()}, skipping batch {micro_batch_idx}"
                )
                continue

            assert self.scaler is not None, "Scaler should be initialized"
            self.scaler.scale(loss).backward()
            performed_backward = True
            accumulated_loss += loss.item()
            successful_steps += 1

        # undo the scaling and get the average loss over successful steps
        if successful_steps > 0:
            self.current_loss = accumulated_loss / successful_steps

        return performed_backward

    def forward_backward(self):
        """ Perform the forward and backward pass, with gradient accumulation if needed. """
        # forward backward update, with gradient accumulation
        assert self.scaler is not None, "Scaler should be initialized"
        performed_backward = self._perform_gradient_accumulation_steps()
        # skip optimizer step if no backward passes were performed
        if not performed_backward:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.update()
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

    def get_qualitative_metrics(self, verbose: bool = True) -> tuple:
        """ Generate random completions and compute the mean perplexity perceived by GPT-2. """
        # list of encodings
        generations_batch = self._generate_random_completions(
            max_new_tokens=50, num_samples=5, temperature=0.8
        )
        if verbose:
            [self.pbar.write(self.decode(g.squeeze().cpu().tolist())) for g in generations_batch]
        mean_pplx = self.evaluator.perplexity(generations_batch, self.decode)
        coherence = self.evaluator.coherence_rate(generations_batch, self.decode)
        token_entropy = self.evaluator.token_entropy(generations_batch)
        return mean_pplx, coherence, token_entropy

    def eval_step(self):
        """Evaluate the model and log results. Save a checkpoint if the model is the best seen so far."""
        self.model.eval()
        losses = self.estimate_loss()
        mean_pplx, coherence, token_entropy = self.get_qualitative_metrics()

        # Update evaluation state
        self.eval_state.update(
            epoch=self.training_state.epoch,
            iter_num=self.training_state.iter_num,
            train_loss=losses["train"],
            val_loss=losses["val"],
            mean_perplexity=mean_pplx,
            coherence_rate=coherence,
            token_entropy=token_entropy,
            lr=self.lr,
            current_loss=self.current_loss
        )

        if self.wandb_logger:
            self.wandb_logger.log(self.eval_state.log_state())
        
        if losses["val"] < self.training_state.best_val_loss or self.config["always_save_checkpoint"]:
            self.training_state.best_val_loss = losses["val"]
            self.eval_state.best_val_loss = losses["val"]
            if self.eval_state.iter_num > 0:
                self.save_checkpoint()

        self.training_state.model.train()

    def training_step(self):
        """Perform a single training step."""

        # determine and set the learning rate for this iteration
        self.update_optimizer_lr(self.training_state.iter_num)

        # at the end of training, we can skip the final forward/backward pass
        if self.training_state.iter_num == 0 and self.config["eval_only"]:
            return

        # else, perform the forward/backward pass and clear memory
        self.forward_backward()

        if self.device_type == "mps":
            cleanup_mps_memory()
        elif self.device_type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def _handle_run_time_error(self, e: RuntimeError) -> bool:
        """
        Inspect a RuntimeError and decide whether to continue training or exit.
        Args:
            e: The RuntimeError encountered during training.
        Returns:
            bool: True if training should continue, False to exit.
        """
        continue_training = False
        if "out of memory" in str(e):
            self.pbar.set_postfix_str("WARNING: ran out of memory, skipping batch")
            time.sleep(1)  # give the GPU a second to recover
            if self.device_type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            elif self.device_type == "mps":
                cleanup_mps_memory()
            continue_training = True
        else: # it is a serious error, exit training
            self.pbar.close()
            print(f"Error during training step: {e}")
            print(str(e))
            print("Memory usage summary:")
            if self.device_type == "cuda":
                print(torch.cuda.memory_summary())
            elif self.device_type == "mps":
                print(get_mps_memory_info())
        return continue_training

    def _generate_random_completions(self, max_new_tokens=200, num_samples=10, temperature=0.8, top_k=200) -> list:
        """ Generate random completions from the model for qualitative evaluation. """
        prompts = ["the", "once upon a time", "the meaning of life is", "the book is", "the capital of france is"]
        completions = []
        contexts = [
            torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device_type)[None, ...]
            for prompt in prompts[:num_samples]
            ]
        assert len(contexts) > 0 and len(contexts) <= num_samples, "invalid number of contexts generated"
        # run generation
        with torch.no_grad():
            with self.ctx:
                for context in contexts:
                    completion = self.training_state.model.generate(
                        context, max_new_tokens, temperature=temperature, top_k=top_k
                    )
                    completions.append(completion)
        return completions
    
    def epoch(self):
        for batch_idx in range(self._batches_per_epoch):
            self._batch_idx = batch_idx
            self.training_state.iter_num += 1
            self.training_state.stage_iter_num += 1
            # evaluate periodically
            if (
                self.training_state.iter_num % self.config["eval_interval"] == 0
                and self.config["master_process"]
                and (self.training_state.iter_num > 0 or self.training_state.epoch > 0)
            ):
                self.eval_step()
            
            start_time = time.time()
            self.training_step()
            end_time = time.time()

            self.pbar.update()
            self.pbar.set_postfix_str(
                f"lr {self.training_state.lr:.2e}, loss {self.current_loss:.4f}, "
                f"tokens {self.training_state.observed_tokens_count:,}"
            )
            self.training_state.batch_process_time = end_time - start_time
            
            if (
                self.training_state.iter_num % self.config["log_interval"] == 0 
                and self.wandb_logger
            ):
                self.wandb_logger.log(self.training_state.log_state())
                self.wandb_logger.log(
                    self.training_state.model.time_consumption.log_state(), step=self.training_state.iter_num
                )
                self.training_state.model.time_consumption.reset()

    def train(self):
        """
        Train the model.
        """
        assert isinstance(self.training_state, TrainingState), "Training state not initialized"
        assert isinstance(self.training_state.model, Transformer), "Model not initialized"
        assert isinstance(self.eval_state, EvaluationState), "Eval state not initialized"

        for epoch in range(self.training_state.epoch, self.config["n_epochs"]):
            self._create_dataloader_indices()
            self.pbar = tqdm(
                total=self._batches_per_epoch,
                initial=self.training_state.iter_num % self._batches_per_epoch,
                desc=f"epoch {epoch+1}/{self.config['n_epochs']}",
            )
            
            try:
                self.epoch()
            
            except KeyboardInterrupt:
                self.save_checkpoint()
                self.pbar.close()
                print("Exiting from training early.")
                break

            except RuntimeError as e:
                continue_training = self._handle_run_time_error(e)
                if not continue_training:
                    print("Exiting from training due to error.")
                    break
            
            # save checkpoint at end of epoch
            if self.config["master_process"]:
                self.training_state.epoch = epoch + 1
                self.save_checkpoint()
            
            self.pbar.clear()

        self.pbar.close()
        print(f"exiting training - epoch {epoch + 1}, iter {self.iter_num}")
