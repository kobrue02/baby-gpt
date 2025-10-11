from datasets import load_dataset

def load_hh_rlhf_dataset():
    ds = load_dataset("Anthropic/hh-rlhf")
    return ds