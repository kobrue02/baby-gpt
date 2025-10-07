"""
The actual implementation of the transformer (BabyGPT) mostly by Karpathy.
Uses multi-head attention with the MHA class from attn.py instead of CausalSelfAttention.
And uses the PackedSwiGLUFFN activation from act.py, instead of the standard GELU.
"""

import torch
import math

from torch import nn, Tensor
from torch.nn import functional as F

from jaxtyping import Float, jaxtyped, Int64
from beartype import beartype as typechecker

from data.utils import enc
from training.configurator import GPTConfig
from training.classes.transformer import Transformer
from training.pretraining.components.blocks import Block, LayerNorm
from training.pretraining.components.muon_optim import MuonEnhanced


import torch
import torch.nn.functional as F

def compute_goldfish_loss(logits, targets, mask_rate=0.02):
    """
    Compute the Goldfish Loss for memorization mitigation.
    
    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        targets: Tensor of shape [batch_size, seq_len]
        mask_rate: Fraction of tokens to include in the loss (default = 2%)
    Returns:
        Scalar tensor for loss.
    """

    # use gpu
    logits = logits.to(targets.device)
    targets = targets.to(logits.device)


    # Flatten logits and targets for simplicity
    batch_size, seq_len, vocab_size = logits.size()
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Create binary mask G ∈ {0,1}^L (1 means include this token)
    # Masking ignores padding tokens (-1)
    valid_mask = (targets_flat != -1)
    random_mask = torch.rand_like(targets_flat.float()) < mask_rate
    goldfish_mask = valid_mask & random_mask  # apply both conditions

    # Compute log probabilities
    log_probs = F.log_softmax(logits_flat, dim=-1)

    # Select the log prob corresponding to the correct target token
    selected_log_probs = log_probs[torch.arange(logits_flat.size(0)), targets_flat]
    
    # Apply mask and compute mean over selected tokens
    masked_log_probs = selected_log_probs[goldfish_mask]
    loss = -masked_log_probs.mean() if masked_log_probs.numel() > 0 else torch.tensor(0.0, device=logits.device)

    return loss




class GPTWithMHA(Transformer):
    """ the full GPT language model, with a context size of block_size """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # n hidden layers
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # type: ignore # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() # type: ignore
        return n_params

    def _init_weights(self, module):
        """ Initialize weights with a near zero normal distribution, and biases to zero. """
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # torch.nn.init.kaiming_normal_(module.weight, a=0.0, mode='fan_in', nonlinearity='silu')
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


    def compute_loss(self, logits, targets):
        """ Compute the cross-entropy loss, ignoring padding tokens (-1). """
        # return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return compute_goldfish_loss(logits, targets, mask_rate=0.02)

    def forward(
        self,
        idx: Int64[torch.Tensor, "batch_size sequence_length"],
        targets: Int64[torch.Tensor, "batch_size sequence_length"] | None = None
        ):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # type: ignore # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # type: ignore # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # type: ignore
        for block in self.transformer.h: # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x) # type: ignore

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = self.compute_loss(logits, targets)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    @jaxtyped(typechecker=typechecker)
    def crop_block_size(self, block_size: int):
        """
        Crop the model block size to the given value.
        """
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size]) # type: ignore
        for block in self.transformer.h: # type: ignore
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    # @jaxtyped(typechecker=typechecker)
    def configure_optimizers(
        self, 
        weight_decay: Float,
        learning_rate: Float,
        betas: tuple[Float, Float],
        device_type
        ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        optimizer = MuonEnhanced(optim_groups, lr=learning_rate, beta1=betas[0], beta2=betas[1], device=device_type)

        return optimizer

    # @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def generate(
        self, 
        idx: Float[Tensor, "batch_size sequence_length"],
        max_new_tokens: int,
        temperature: Float = 1.0,
        top_k: int | None = None
        ) -> Float[Tensor, "batch_size new_sequence_length"]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # stop generating if we hit the end of text token
            if int(idx_next[0, 0]) == enc.eot_token:
                break
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
