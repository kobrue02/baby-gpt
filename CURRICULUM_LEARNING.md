# Curriculum Learning for Pretraining

This implementation provides a curriculum learning approach for pretraining, where the model progressively learns from easier data with shorter contexts to more complex data with longer contexts.

## Overview

Curriculum learning trains models through multiple stages:
1. **Warmup**: Simple, short texts (e.g., TinyStories) with small block size
2. **Foundation**: More complex data with medium block size
3. **Pretrain**: Full complexity data with large block size

Each stage has:
- Different dataset (progressively more complex)
- Different block size (progressively longer contexts)
- Different number of epochs (progressively more training)

## File Structure

- `data_loaders/curriculum.py` - Defines `CurriculumStage` and `Curriculum` dataclasses
- `training/pretraining/curriculum_trainer.py` - `CurriculumTrainer` class that handles stage transitions
- `data_loaders/utils.py` - Updated to support stage-based file naming
- `main.py` - CLI commands for curriculum learning

## Binary File Naming Convention

Datasets for curriculum learning are saved with stage names:
```
data/train_{stage}_{suffix}.bin
data/val_{stage}_{suffix}.bin
```

Examples:
- `train_warmup_pretrain.bin` - Training data for warmup stage
- `val_foundation_pretrain.bin` - Validation data for foundation stage
- `train_pretrain_pretrain.bin` - Training data for final pretrain stage

## Usage

### 1. Initialize Curriculum Datasets

First, download and prepare all datasets for the curriculum:

```bash
# Simple 2-stage curriculum (warmup + pretrain)
python main.py initialize curriculum --type simple

# Default 3-stage curriculum (warmup + foundation + pretrain)
python main.py initialize curriculum --type default
```

This will:
- Download datasets for each stage
- Tokenize and save them with stage-specific names
- Create both train and validation splits

### 2. Start Curriculum Training

Train from scratch using curriculum learning:

```bash
# Train with simple curriculum
python main.py start curriculum --type simple

# Train with default curriculum
python main.py start curriculum --type default
```

### 3. Resume Curriculum Training

Resume from a checkpoint (automatically continues from the correct stage):

```bash
# Resume simple curriculum
python main.py resume curriculum --type simple

# Resume default curriculum
python main.py resume curriculum --type default
```

## Curriculum Types

### Simple (2 stages)
- **Warmup**: TinyStories, 128 block size, 1 epoch, 10k examples
- **Pretrain**: Custom web scrape (URLs from db.py), 256 block size, 2 epochs

### Default (3 stages)
- **Warmup**: TinyStories, 128 block size, 1 epoch, 50k examples
- **Foundation**: Local .txt files (from `data_loaders/files/`), 256 block size, 2 epochs
- **Pretrain**: Custom web scrape (URLs from db.py), 512 block size, 3 epochs

## Dataset Sources

The curriculum uses three types of data sources:

1. **HuggingFace datasets** (e.g., TinyStories): Simple children's stories for warmup
2. **Local text files** (`local-files`): Your custom .txt files in `data_loaders/files/`
3. **Web-scraped content** (`custom-scrape`): URLs, PDFs, and blog posts from `data_loaders/db.py`

The web scraper (`ScrapedDataLoader`) automatically excludes local files in the pretrain stage, so they're only used in the foundation stage.

## Creating Custom Curriculums

You can define custom curriculums in code:

```python
from data_loaders.curriculum import Curriculum, CurriculumStage

curriculum = Curriculum(stages=[
    CurriculumStage(
        name="warmup",
        dataset_key="roneneldan/TinyStories",
        block_size=128,
        n_epochs=1,
        dataset_suffix="pretrain",
        n_items=50000
    ),
    CurriculumStage(
        name="main",
        dataset_key="HuggingFaceFW/fineweb-edu",
        block_size=512,
        n_epochs=3,
        dataset_suffix="pretrain",
        n_items=None  # Use full dataset
    ),
])
```

## How It Works

### Stage Transitions

The `CurriculumTrainer` automatically handles stage transitions:

1. **Trains** for the specified number of epochs in the current stage
2. **Saves** a checkpoint with stage information
3. **Transitions** to the next stage by:
   - Updating block size
   - Loading the new dataset files
   - Recalculating training parameters
   - Optionally reinitializing the learning rate scheduler
4. **Resumes** training with the new configuration

### Checkpoint Format

Checkpoints save curriculum progress:
```python
{
    'curriculum_stage_idx': 1,  # Current stage index
    'epoch': 0,                  # Epoch within current stage
    'iter_num': 1000,            # Total iterations
    # ... other training state ...
}
```

### Dataset Loading

During training, the trainer loads stage-specific datasets:
```python
# For stage "warmup", loads:
train_data = "data/train_warmup_pretrain.bin"
val_data = "data/val_warmup_pretrain.bin"

# For stage "foundation", loads:
train_data = "data/train_foundation_pretrain.bin"
val_data = "data/val_foundation_pretrain.bin"
```

## Benefits of Curriculum Learning

1. **Faster convergence**: Model learns basic patterns on simple data first
2. **Better generalization**: Progressive difficulty prevents overfitting
3. **Efficient training**: Short contexts in early stages = faster training
4. **Flexible**: Can adjust difficulty curve based on your needs

## Example Training Flow

```bash
# Step 1: Initialize datasets
python main.py initialize curriculum --type simple

# Output:
# [Stage 1/2] warmup
#   Dataset: roneneldan/TinyStories
#   Block size: 128
#   Epochs: 1
#   Items: 10,000
#   ✓ Stage 'warmup' dataset created successfully!
#
# [Stage 2/2] pretrain
#   Dataset: roneneldan/TinyStories
#   Block size: 256
#   Epochs: 2
#   Items: 50,000
#   ✓ Stage 'pretrain' dataset created successfully!

# Step 2: Start training
python main.py start curriculum --type simple

# Training progresses through:
# Stage 1: warmup (128 block size, 1 epoch)
# Stage 2: pretrain (256 block size, 2 epochs)

# Step 3: If interrupted, resume
python main.py resume curriculum --type simple
# Automatically continues from the correct stage
```

## Configuration

The curriculum trainer uses the same configuration as regular pretraining (from `training/configurator.py`), but overrides:
- `block_size`: Set per stage
- `n_epochs`: Set per stage

All other hyperparameters (learning rate, batch size, etc.) remain consistent across stages.

## Tips

1. **Start small**: Test with `--type simple` and small `n_items` first
2. **Monitor transitions**: Watch logs to see stage transitions
3. **Adjust block sizes**: Match your model's capacity and memory limits
4. **Progressive epochs**: Train longer on more complex stages
5. **Dataset quality**: Use progressively higher-quality or more diverse data

## Troubleshooting

**Error: Dataset file not found**
```
FileNotFoundError: Dataset file not found: data/train_warmup_pretrain.bin
```
→ Run `python main.py initialize curriculum --type [simple|default]` first

**Stage not resuming correctly**
→ Make sure you use the same `--type` when resuming as when you started

**Memory issues**
→ Reduce block size in earlier stages or decrease batch size in config
