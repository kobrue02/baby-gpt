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
    
    def perplexity(self, input_texts: List[str]) -> float:
        encodings = self.tokenizer(input_texts, return_tensors="pt")

        # GPT-2 needs both inputs and labels to compute loss
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = math.exp(loss.item())

        return perplexity

