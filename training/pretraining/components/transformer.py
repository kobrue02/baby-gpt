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
from training.pretraining.components.muon_optim import SingleDeviceMuonWithAuxAdam
from training.pretraining.components.loss import compute_goldfish_loss
from training.pretraining.components.yarn import LlamaYaRNScaledRotaryEmbedding


import torch
import torch.nn.functional as F


class GPTWithMHA(Transformer):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.wpe = nn.Embedding(
            0, 0
        )  # avoid attribute error if not using rotary embeddings
        if not config.use_rotary:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            print("using rotary YaRN embeddings")
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # n hidden layers
                ln_f=LayerNorm(config.n_embd, bias=config.bias),  # final layer norm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # type: ignore # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rotary:
            n_params -= self.wpe.weight.numel()  # type: ignore
        return n_params

    def _init_weights(self, module):
        """Initialize weights with a near zero normal distribution, and biases to zero."""
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # torch.nn.init.kaiming_normal_(module.weight, a=0.0, mode='fan_in', nonlinearity='silu')
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def compute_loss(self, logits, targets):
        """Compute the cross-entropy loss, ignoring padding tokens (-1)."""
        # return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return compute_goldfish_loss(logits, targets)

    def forward(
        self,
        idx: Int64[torch.Tensor, "batch_size sequence_length"],
        targets: Int64[torch.Tensor, "batch_size sequence_length"] | None = None,
    ):
        device = idx.device
        _, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if not self.config.use_rotary:
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore
        else:
            x = self.transformer.drop(tok_emb)  # type: ignore

        for block in self.transformer.h:  # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x)  # type: ignore

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = self.compute_loss(logits, targets)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
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
        if not self.config.use_rotary:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])  # type: ignore
        for block in self.transformer.h:  # type: ignore
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # @jaxtyped(typechecker=typechecker)
    def configure_optimizers(
        self,
        weight_decay: Float,
        learning_rate: Float,
        betas: tuple[Float, Float],
        device_type,
    ):

        # acc to muon paper
        hidden_weights = [
            p for p in self.transformer.parameters() if p.ndim >= 2 and p.requires_grad
        ]
        hidden_gains_biases = [
            p for p in self.transformer.parameters() if p.ndim < 2 and p.requires_grad
        ]
        nonhidden_params = (
            [*self.lm_head.parameters(), *self.wpe.parameters()]
            if not self.config.use_rotary
            else [*self.lm_head.parameters()]
        )
        nonhidden_params = [p for p in nonhidden_params if p.requires_grad]

        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=learning_rate,
                weight_decay=weight_decay,
            ),
            dict(
                params=hidden_gains_biases + nonhidden_params,
                use_muon=False,
                lr=learning_rate,
                betas=betas,
                weight_decay=0.0,
            ),
        ]

        print(
            f"training {int(len(hidden_weights)+len(hidden_gains_biases)+len(nonhidden_params))} trainable parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        return optimizer

    # @jaxtyped(typechecker=typechecker)
    @torch.no_grad()
    def generate(
        self,
        idx: Float[Tensor, "batch_size sequence_length"],
        max_new_tokens: int,
        temperature: Float = 1.0,
        top_k: int | None = None,
    ) -> Float[Tensor, "batch_size new_sequence_length"]:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
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
