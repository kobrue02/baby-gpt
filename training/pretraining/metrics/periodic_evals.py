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
    
    def perplexity(self, encodings_batch: List[torch.Tensor], decoder: Callable) -> float:
        """
        Calculate mean perplexity across a batch of encoded sequences.

        Args:
            encodings_batch: List of tensors containing token IDs

        Returns:
            Mean perplexity across all sequences
        """
        perplexities = []

        with torch.no_grad():
            for encoding in encodings_batch:
                try:
                    text: str = decoder(encoding.squeeze().cpu().tolist())
                except KeyError:
                    continue
                inputs = self.tokenizer(text, return_tensors="pt")
                input_ids = inputs.input_ids

                # GPT-2 needs both inputs and labels to compute loss
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = math.exp(loss.item())
                perplexities.append(perplexity)

        if len(perplexities) == 0:
            return float('inf')
        
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