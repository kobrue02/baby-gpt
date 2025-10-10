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
    
    def perplexity(self, encodings_batch: List[torch.Tensor]) -> float:
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
                # Convert tensor to correct format if needed
                if isinstance(encoding, torch.Tensor):
                    # Remove batch dimension if present and convert to CPU
                    input_ids = encoding.squeeze().cpu()
                    # Add batch dimension back for model input
                    input_ids = input_ids.unsqueeze(0)
                else:
                    # If it's already a list of token IDs
                    input_ids = torch.tensor([encoding])

                # GPT-2 needs both inputs and labels to compute loss
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = math.exp(loss.item())
                perplexities.append(perplexity)

        return sum(perplexities) / len(perplexities)

