"""
Sample from a trained model
"""
import os
import pickle
import argparse
import torch
import tiktoken

from contextlib import nullcontext

from training.components.transformer import GPTWithMHA
from training.components.blocks import GPTConfig
from training.configurator import load_config

# -----------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate text from a trained model')
parser.add_argument('--prompt', type=str, required=True, help='The prompt to complete')
parser.add_argument('--init_from', type=str, default='resume', help='Either "resume" (from an out_dir) or a gpt2 variant')
parser.add_argument('--out_dir', type=str, default='out', help='Output directory for checkpoints')
parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to draw')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling parameter')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--compile', action='store_true', help='Use PyTorch 2.0 compilation')
args = parser.parse_args()

# Load configuration
config = load_config()

# Set variables from args and config
init_from = args.init_from
out_dir = args.out_dir
start = args.prompt
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
compile_model = args.compile

# Device detection with MPS support
if torch.cuda.is_available():
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    dtype = 'float32'  # MPS doesn't support bfloat16 well
else:
    device = 'cpu'
    dtype = 'float32'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.fp32_precision = 'tf32' # use tf32 for matmul
torch.backends.cudnn.conv.fp32_precision = 'tf32' # use tf32 for cudnn # type: ignore
device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type in ['cpu', 'mps'] else torch.amp.autocast(device_type=device_type, dtype=ptdtype) # type: ignore

# model
def load_model(out_dir, device, compile_model=True):
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract model configuration from checkpoint
    model_config = checkpoint['config']
    model_args = dict(
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        block_size=model_config['block_size'],
        bias=model_config['bias'],
        dropout=0.0,  # Set to 0 for inference
        vocab_size=model_config.get('vocab_size', 50304)
    )

    gptconf = GPTConfig(**model_args)
    model = GPTWithMHA(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    return model, checkpoint

# look for the meta pickle in case it is available in the dataset folder
def encode_decode(init_from, checkpoint):
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    return encode, decode

# encode the beginning of the prompt
def generate(model, start: str, encode, decode):
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')


if __name__ == '__main__':
    model, checkpoint = load_model(out_dir, device, compile_model)
    encode, decode = encode_decode(init_from, checkpoint)
    generate(model, start, encode, decode)