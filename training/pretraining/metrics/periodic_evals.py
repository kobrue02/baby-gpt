from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Any
import torch
import math


class PeriodicEval:
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
        self.model.eval()
    
    def perplexity(self, encodings) -> float:
        # Convert tiktoken encoding (list of token IDs) to tensor
        input_ids = torch.tensor([encodings])  # Add batch dimension

        # GPT-2 needs both inputs and labels to compute loss
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        return perplexity

