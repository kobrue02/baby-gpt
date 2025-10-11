# LLM Training

Tiny GPT implementation based on Karpathy's nanoGPT code using Goldfish loss and Muon optimizer.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The CLI provides commands for both pretraining and supervised fine-tuning (SFT):

### Dataset Preparation

The data loaders use **streaming mode by default** to save disk space by not caching datasets. Data is downloaded, tokenized, and written directly to binary files in one pass.

#### Pretraining Datasets

```bash
# List available datasets
python main.py initialize pretraining --list

# Use a preset dataset (streaming mode - saves disk space)
python main.py initialize pretraining --dataset fineweb --n-items 10000

# Use a custom HuggingFace dataset
python main.py initialize pretraining --dataset-key HuggingFaceFW/fineweb --n-items 5000

# Use a specific subset
python main.py initialize pretraining --dataset smollm-corpus --subset python-edu --n-items 10000

# Disable streaming (caches dataset to disk)
python main.py initialize pretraining --dataset fineweb --n-items 10000 --no-streaming
```

**Available preset datasets:**
- `fineweb` - High-quality web text from Common Crawl
- `fineweb-edu` - Educational subset of FineWeb
- `smollm-corpus` - SmolLM pretraining corpus (cosmopedia-v2)
- `cosmopedia` - Synthetic textbooks and educational content
- `python-edu` - Educational Python code
- `web-samples` - Deduplicated web samples

#### SFT Datasets

```bash
# List available datasets
python main.py initialize sft --list

# Use default dataset (streaming mode)
python main.py initialize sft --n-items 10000

# Use a custom dataset
python main.py initialize sft --dataset-key username/my-dataset --n-items 5000

# Disable streaming (caches dataset to disk)
python main.py initialize sft --n-items 10000 --no-streaming
```

**Available preset datasets:**
- `general-knowledge` - General knowledge Q&A pairs

### Training

```bash
# Start pretraining from scratch
python main.py start pretraining

# Resume pretraining from checkpoint
python main.py resume pretraining

# Start SFT
python main.py start sft

# Resume SFT from checkpoint
python main.py resume sft
```

### Generation

```bash
# Generate text from a prompt
python main.py generate "Once upon a time" --max-tokens 100 --temperature 0.8

# Use SFT checkpoint
python main.py generate "What is the capital of France?" --sft --max-tokens 50

# Interactive mode
python main.py interactive --sft --temperature 0.7
```

### Management

```bash
# Check training status
python main.py status

# Clean up checkpoints
python main.py clean pretraining
python main.py clean sft
```

## Help

For detailed options on any command:

```bash
python main.py [command] --help
```
