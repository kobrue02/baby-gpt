import evaluate

from typing import List

class PeriodicEval:
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.pplx = evaluate.load("perplexity", module_type="metric")
    
    def perplexity(self, input_texts: List[str]) -> float:
        """ Compute the perplexity of the model on the given input texts. """
        results = self.pplx.compute(model_id=self.model_id,
                                    add_start_token=False,
                                    predictions=input_texts)
        if not results:
            raise ValueError("No results returned from perplexity evaluation.")
        
        mean_pplx = results['mean_perplexity']
        return mean_pplx
