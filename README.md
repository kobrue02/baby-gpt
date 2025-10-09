# LLM Training

Tiny GPT implementation based on Karpathy's nanoGPT code using Goldfish loss and Muon optimizer.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The CLI provides commands for both pretraining and supervised fine-tuning (SFT):

### Dataset Preparation

```bash
# Initialize pretraining dataset
python main.py initialize-pretraining --n-shards 10 --dataset facebook/recycling_the_web

# Initialize SFT dataset
python main.py initialize-sft --n-rows 10000
```

### Training

```bash
# Start pretraining from scratch
python main.py start-pretraining

# Resume pretraining from checkpoint
python main.py resume-pretraining

# Start SFT
python main.py start-sft

# Resume SFT from checkpoint
python main.py resume-sft
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
