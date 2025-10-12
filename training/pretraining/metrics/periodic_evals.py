from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Callable
import torch
import math
import re
from collections import Counter
import numpy as np


class PeriodicEval:
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
        self.model.eval()
    
    def perplexity(self, encodings_batch: List[torch.Tensor]) -> float:
        """
        Calculate mean perplexity across a batch of encoded sequences.

        Args:
            encodings_batch: List of tensors containing token IDs

        Returns:
            Mean perplexity across all sequences
        """
        perplexities = []
        vocab_size = self.model.config.vocab_size  # GPT-2 vocab size (50257)

        with torch.no_grad():
            for encoding in encodings_batch:
                # Convert tensor to correct format if needed
                if isinstance(encoding, torch.Tensor):
                    # Remove batch dimension if present and convert to CPU
                    input_ids = encoding.squeeze().cpu()
                    # Add batch dimension back for model input
                    input_ids = input_ids.unsqueeze(0)
                else:
                    # If it's already a list of token IDs
                    input_ids = torch.tensor([encoding])

                # Clamp token IDs to valid vocab range
                # Replace out-of-vocab tokens with unknown token (typically 0 or vocab_size-1)
                input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

                # GPT-2 needs both inputs and labels to compute loss
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = math.exp(loss.item())
                perplexities.append(perplexity)

        return sum(perplexities) / len(perplexities)

    def coherence_rate(self, encodings_batch: List[torch.Tensor], decode_fn: Callable) -> float:
        """% of generations ending with proper punctuation (. ! ?)"""
        valid = sum(1 for enc in encodings_batch 
                   if re.search(r'[.!?]\s*$', decode_fn(enc.squeeze().cpu().tolist()).strip()))
        return valid / len(encodings_batch)
    
    def token_entropy(self, encodings_batch: List[torch.Tensor]) -> float:
        """Mean Shannon entropy across sequences"""
        entropies = []
        for enc in encodings_batch:
            tokens = enc.squeeze().cpu().tolist()
            counts = Counter(tokens)
            probs = [c / len(tokens) for c in counts.values()]
            entropies.append(-sum(p * math.log(p) for p in probs if p > 0))
        return np.mean(entropies).astype(float)