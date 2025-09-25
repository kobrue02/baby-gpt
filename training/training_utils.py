"""
This training script can be run on a single gpu in debug mode.

To run on a single GPU, example:
$ python -m training.train
"""

import os
import math
import pickle
import numpy as np
import torch

from tqdm import tqdm
from training.components.transformer import GPTWithMHA
from training.components.blocks import GPTConfig
from training.configurator import load_config


def get_mps_memory_info():
    """Helper function to monitor MPS memory usage"""
    import torch
    if torch.backends.mps.is_available():
        return {
            'allocated': torch.mps.current_allocated_memory() / 1024**3,  # GB
            'reserved': torch.mps.driver_allocated_memory() / 1024**3     # GB
        }
    return None


def cleanup_mps_memory():
    """Clean up MPS memory between training steps if needed"""
    import torch
    import gc
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        gc.collect()


class Trainer:
    """
    Trainer class to handle model training.
    """
    def __init__(self):
        self.config: dict
        self.latest_checkpoint: dict
        self.best_val_loss: float
        self.X: torch.Tensor
        self.Y: torch.Tensor
        self.seen_batches = set()
        self.lr: float

        self.load_and_validate_config()
        self.device_type, self.ptdtype, self.ctx = self.setup_device()
        self.meta_vocab_size, _, _ = self.derive_vocab_size('data')

        # initialize model
        self.model = self.init_model(self.meta_vocab_size, self.config.get('compile', True))
        self.optimizer, self.scaler = self.init_optimizer_and_scaler()

        self.wandb_logger = self.setup_logging()


    def load_and_validate_config(self):
        """
        Load and validate the training configuration.
        """
        config = load_config()

        # validate config values
        assert config['device'] in ['cpu', 'cuda', 'mps'], "Only cpu, cuda, and mps devices are supported in this script."
        assert config["n_embd"] % config["n_head"] == 0, "Embedding size must be divisible by number of heads."

        print(f"tokens per iteration will be: {config['tokens_per_iter']:,}")
        self.config = config

    def setup_device(self):
        """
        Setup the device and context to use for training.
        """
        if self.config["master_process"]:
            os.makedirs(self.config["out_dir"], exist_ok=True)
        torch.manual_seed(1337 + self.config["seed_offset"])
        torch.backends.cuda.matmul.fp32_precision = 'tf32' # use tf32 for matmul
        torch.backends.cudnn.conv.fp32_precision = 'tf32' # type: ignore
        device_type = self.config["device"]
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config["dtype"]]
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) # type: ignore
        return device_type, ptdtype, ctx
    
    def find_unseen_batch(self, data: np.memmap):
        """
        Find a new unseen batch of data indices.
        """
        ix = torch.randint(len(data) - self.config["block_size"], (self.config["batch_size"],))

        # Convert tensor to tuple for set membership check
        ix_tuple = tuple(ix.tolist())
        max_attempts = 5
        attempts = 0

        while ix_tuple in self.seen_batches and attempts < max_attempts:
            ix = torch.randint(len(data) - self.config["block_size"], (self.config["batch_size"],))
            ix_tuple = tuple(ix.tolist())
            attempts += 1

        # Reset seen_batches if we've seen too many (prevent memory growth)
        if len(self.seen_batches) > 10000:
            self.seen_batches.clear()

        self.seen_batches.add(ix_tuple)
        return ix

    def get_batch(self, split: str, data_dir='data'):
        """
        Generate a batch of data of inputs x and targets y with np.memmap
        """
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        ix = self.find_unseen_batch(data)
        
        x = torch.stack([torch.from_numpy((data[i:i+self.config["block_size"]]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.config["block_size"]]).astype(np.int64)) for i in ix])
        
        x, y = x.to(self.device_type), y.to(self.device_type)
        return x, y


    def derive_vocab_size(self, data_dir):
        """
        Attempt to derive the vocab size from the dataset's meta.pkl file.
        """
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        
        return meta_vocab_size, iter_num, best_val_loss

    def init_model(self, meta_vocab_size, compile_model=True):
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
            dropout=self.config["dropout"]
            )
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPTWithMHA(gptconf)

        if compile_model:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model) # requires PyTorch 2.0
        
        return model.to(self.device_type)

    def init_optimizer_and_scaler(self):
        """ Initialize the optimizer and gradient scaler. """
        scaler = torch.amp.GradScaler(device=self.device_type, enabled=(self.config['dtype'] == 'float16')) # type: ignore
        optimizer = self.model.configure_optimizers(
            self.config['weight_decay'],
            self.config['learning_rate'],
            (self.config['beta1'], self.config['beta2']),
            self.device_type
        )
        return optimizer, scaler


    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, data_dir='data'):
        """
        Estimate the loss over either split using many batches, so that we get a more accurate estimate
        """
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config['eval_iters'])
            for k in range(self.config['eval_iters']):
                self.X, self.Y = self.get_batch(split, data_dir)
                with self.ctx:
                    _, loss = self.model(self.X, self.Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.config['warmup_iters']:
            return self.config['learning_rate'] * (it + 1) / (self.config['warmup_iters'] + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config['lr_decay_iters']:
            return self.config['min_lr']
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config['warmup_iters']) / (self.config['lr_decay_iters'] - self.config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config['min_lr'] + coeff * (self.config['learning_rate'] - self.config['min_lr'])

    def setup_logging(self):
        if self.config['wandb_log'] and self.config['master_process']:
            import wandb
            wandb.init(project=self.config['wandb_project'], name=self.config['wandb_run_name'], config=self.config)
            return wandb
        return None
    
    def update_optimizer_lr(self, iter_num):
        """ Get and set the learning rate for the current iteration """
        self.lr = self.get_lr(iter_num) if self.config['decay_lr'] else self.config['learning_rate']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def forward_backward(self, data_dir='data'):
        """
        Perform the forward and backward pass, with gradient accumulation if needed.
        """
        # forward backward update, with optional gradient accumulation
        for _ in range(self.config['gradient_accumulation_steps']):
            with self.ctx:
                _, loss = self.model(self.X, self.Y)
                loss = loss / self.config['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()
            # get a new random unseen batch
            self.X, self.Y = self.get_batch('train', data_dir)

        # clip the gradient
        if self.config['grad_clip'] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

        # step the optimizer and scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(self, iter_num):
        checkpoint = {
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        print(f"saving checkpoint to {self.config['out_dir']}")
        torch.save(checkpoint, os.path.join(self.config['out_dir'], 'ckpt.pt'))
        self.latest_checkpoint = checkpoint

    def eval_step(self, data_dir='data', iter_num=0):
        """ Evaluate the model and log results. Save a checkpoint if the model is the best seen so far. """
        losses = self.estimate_loss(data_dir)
        if self.wandb_logger:
            self.wandb_logger.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": self.lr,
            })
        if losses['val'] < self.best_val_loss or self.config['always_save_checkpoint']:
            self.best_val_loss = losses['val']
            if iter_num > 0:
                self.save_checkpoint(iter_num)

    def training_step(self, iter_num=0, data_dir='data'):
        """ Perform a single training step. """
        self.pbar.update()

        # determine and set the learning rate for this iteration
        self.update_optimizer_lr(iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % self.config['eval_interval'] == 0 and self.config['master_process']:
            self.eval_step(data_dir, iter_num)
        
        # at the end of training, we can skip the final forward/backward pass
        if iter_num == 0 and self.config['eval_only']:
            return

        # else, perform the forward/backward pass and clear memory
        self.forward_backward(data_dir)
        if self.device_type == 'mps':
            cleanup_mps_memory()


    def train(self, data_dir='data'):
        """
        Train the model.
        """

        self.best_val_loss = 1e9
        self.X, self.Y = self.get_batch('train', data_dir)
        self.raw_model = self.model
        self.pbar = tqdm(total=self.config['max_iters'])

        iter_num = 0
        while iter_num < self.config['max_iters']:
            try:
                self.training_step(iter_num, data_dir)
                iter_num += 1
            except KeyboardInterrupt:
                self.save_checkpoint(iter_num)
                print("Exiting from training early.")
                break