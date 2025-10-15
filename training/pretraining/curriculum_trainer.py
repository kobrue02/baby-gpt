"""
Curriculum Trainer for progressive pretraining.

This trainer implements curriculum learning by training the model through
multiple stages with progressively:
- More complex data
- Longer context (block_size)
- More training epochs
"""

import os
import numpy as np
import torch
from typing import Optional

from training.pretraining.pretrainer import PreTrainer
from data_loaders.curriculum import Curriculum, CurriculumStage


class CurriculumTrainer(PreTrainer):
    """
    Trainer that implements curriculum learning with progressive difficulty.

    The model trains through multiple stages, each with different:
    - Dataset (easy -> hard)
    - Block size (short -> long context)
    - Number of epochs (few -> many)
    """

    def __init__(self, curriculum: Curriculum, resume: bool = False, data_dir: str = "data"):
        """
        Initialize the curriculum trainer.

        Args:
            curriculum: Curriculum object defining the training stages
            resume: whether to resume from a checkpoint
            data_dir: directory containing the binary datasets
        """
        self.curriculum = curriculum
        self.current_stage_idx = 0
        self.current_stage: Optional[CurriculumStage] = None

        # Initialize with the first stage's configuration
        self._apply_stage_config(curriculum.stages[0])

        # Call parent init after setting up stage config
        super().__init__(resume=resume, data_dir=data_dir)

        # Override some settings for curriculum learning
        self._setup_curriculum_state()

    def _apply_stage_config(self, stage: CurriculumStage):
        """
        Apply a curriculum stage's configuration to the trainer config.

        This updates the model's block_size and n_epochs for the current stage.
        Must be called before super().__init__() or when transitioning stages.
        """
        self.current_stage = stage

        # Update the config with stage-specific parameters
        # Note: This modifies self.config which is loaded in load_and_validate_config
        if hasattr(self, 'config'):
            self.config["block_size"] = stage.block_size
            self.config["n_epochs"] = stage.n_epochs
            print(f"\nApplying curriculum stage: {stage.name}")
            print(f"  Block size: {stage.block_size}")
            print(f"  Epochs: {stage.n_epochs}")
            print(f"  Dataset: {stage.dataset_key}")

    def _setup_curriculum_state(self):
        """Setup curriculum-specific state after initialization."""
        # Initialize curriculum stage index in training state
        self.training_state.curriculum_stage_idx = self.current_stage_idx

    def _get_stage_dataset_files(self, split: str) -> str:
        """
        Get the dataset file path for the current curriculum stage.

        Args:
            split: 'train' or 'val'

        Returns:
            Path to the binary file for this stage
        """
        assert self.current_stage is not None, "Current stage not set"

        stage_name = self.current_stage.name
        suffix = self.current_stage.dataset_suffix

        # Format: {split}_{stage}_{suffix}.bin
        # E.g., train_warmup_pretrain.bin
        filename = f"{split}_{stage_name}_{suffix}.bin"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Make sure you've created the dataset for stage '{stage_name}' "
                f"using the data loader with stage='{stage_name}'"
            )

        return filepath

    def _get_dataset_length(self, split: str) -> int:
        """Get the length of the dataset for the current stage."""
        filepath = self._get_stage_dataset_files(split)
        data = np.memmap(filepath, dtype=np.uint16, mode="r")
        return len(data)

    def get_batch(self, split: str, indices: torch.Tensor, batch_idx: int):
        """
        Get a batch of data for the current curriculum stage.

        Args:
            split: 'train' or 'val'
            indices: Shuffled indices for this epoch
            batch_idx: Current batch index within the epoch
        """
        # Load from stage-specific files
        filepath = self._get_stage_dataset_files(split)
        data = np.memmap(filepath, dtype=np.uint16, mode="r")

        # Get batch_size indices from the shuffled index array
        start_idx = batch_idx * self.config["batch_size"]
        end_idx = min(start_idx + self.config["batch_size"], len(indices))
        batch_indices = indices[start_idx:end_idx]

        # Load sequences at these positions
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + self.config["block_size"]]).astype(np.int64)
                )
                for i in batch_indices
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.config["block_size"]]).astype(np.int64)
                )
                for i in batch_indices
            ]
        )

        return x.to(self.device_type), y.to(self.device_type)

    def _transition_to_next_stage(self) -> bool:
        """
        Transition to the next curriculum stage.

        Returns:
            True if transitioned to a new stage, False if curriculum is complete
        """
        assert self.current_stage_idx is not None, "Current stage index not set"
        assert self.current_stage is not None, "Current stage not set"
        assert isinstance(self.curriculum, Curriculum), "Curriculum not set"
        assert isinstance(self.current_stage, CurriculumStage), "Current stage invalid"
        self.current_stage_idx += 1

        if self.current_stage_idx >= len(self.curriculum.stages):
            print("\nCurriculum complete! All stages finished.")
            return False

        # Get the next stage
        next_stage = self.curriculum.stages[self.current_stage_idx]

        print(f"\n{'='*60}")
        print(f"TRANSITIONING TO NEXT CURRICULUM STAGE")
        print(f"{'='*60}")
        print(f"Previous stage: {self.current_stage.name}")
        print(f"Next stage: {next_stage.name}")
        print(f"Stage {self.current_stage_idx + 1}/{len(self.curriculum.stages)}")

        # Apply the new stage's configuration
        self._apply_stage_config(next_stage)

        # Recalculate training parameters for the new stage
        self.train_data_len = self._get_dataset_length("train")
        self.steps_per_epoch = self.train_data_len // (
            self.config["batch_size"] * self.config["block_size"]
        )
        self.total_steps = self.steps_per_epoch * self.config["n_epochs"]

        # Recalculate batches per epoch for the new stage
        self._calculate_batches_per_epoch()
        self._validate_and_adjust_intervals()

        # Reset epoch counter and stage iteration counter for this stage
        self.training_state.epoch = 0
        self.training_state.stage_iter_num = 0
        self.training_state.curriculum_stage_idx = self.current_stage_idx

        # Optionally: reinitialize the learning rate schedule for the new stage
        if self.training_state.scheduler is not None:
            self.training_state.scheduler = self.init_scheduler(
                self.training_state.optimizer, self.total_steps
            )

        print(f"New total steps: {self.total_steps}")
        print(f"New steps per epoch: {self.steps_per_epoch}")
        print(f"{'='*60}\n")

        return True

    def train(self):
        """
        Train the model through all curriculum stages.

        This overrides the parent train() method to handle multiple stages.
        """
        print(f"\nStarting curriculum training with {len(self.curriculum.stages)} stages")
        print("=" * 60)
        
        assert self.current_stage_idx is not None, "Current stage index not set"
        assert self.current_stage is not None, "Current stage not set"
        assert isinstance(self.curriculum, Curriculum), "Curriculum not set"
        assert isinstance(self.current_stage, CurriculumStage), "Current stage invalid"
        
        for stage_idx, stage in enumerate(self.curriculum.stages):
            # Skip stages we've already completed (when resuming)
            if stage_idx < self.current_stage_idx:
                print(f"Skipping already completed stage: {stage.name}")
                continue

            print(f"\nStage {stage_idx + 1}/{len(self.curriculum.stages)}: {stage.name}")
            print(f"  Dataset: {stage.dataset_key}")
            print(f"  Block size: {stage.block_size}")
            print(f"  Epochs: {stage.n_epochs}")
            print("-" * 60)

            # Train for this stage using parent's train method
            # But only train for this stage's epochs
            stage_end_epoch = stage.n_epochs

            for epoch in range(self.training_state.epoch, stage_end_epoch):
                try:
                    self._create_dataloader_indices()
                except ValueError as e:
                    print(f"Error creating dataloader indices: {e}")
                    print("Skipping to next stage.")
                    break

                from tqdm import tqdm
                self.pbar = tqdm(
                    total=self._batches_per_epoch,
                    initial=self.training_state.stage_iter_num % self._batches_per_epoch,
                    desc=f"stage {stage.name} | epoch {epoch+1}/{stage_end_epoch}",
                )

                try:
                    self.epoch()

                except KeyboardInterrupt:
                    self.save_checkpoint()
                    self.pbar.close()
                    print("Exiting from training early.")
                    return

                except RuntimeError as e:
                    continue_training = self._handle_run_time_error(e)
                    if not continue_training:
                        print("Exiting from training due to error.")
                        return
                
                except AttributeError as e:
                    print(f"AttributeError during stage {stage.name}, epoch {epoch+1}: {e}")
                    print("Skipping to next stage.")
                    break

                # Save checkpoint at end of epoch
                if self.config["master_process"]:
                    self.training_state.epoch = epoch + 1
                    self.save_checkpoint()

                self.pbar.clear()

            self.pbar.close()

            # Transition to next stage (if not the last stage)
            if stage_idx < len(self.curriculum.stages) - 1:
                if not self._transition_to_next_stage():
                    break  # Curriculum complete

        print(f"\nCurriculum training complete!")
        print(f"Final stage: {self.current_stage.name}")
        print(f"Total iterations: {self.training_state.iter_num}")

    def save_checkpoint(self):
        """Save checkpoint with curriculum stage information."""
        assert self.current_stage is not None, "Current stage not set"

        # Get checkpoint dict and add curriculum stage info
        checkpoint_dict = self.training_state.to_checkpoint_dict()
        checkpoint_dict['curriculum_stage_idx'] = self.current_stage_idx

        try:
            self.pbar.set_postfix_str(
                f"saving checkpoint (stage {self.current_stage.name}) to {self.config['out_dir']}"
            )
        except AttributeError:
            pass

        self._atomic_save_checkpoint(checkpoint_dict)

    def load_checkpoint(self):
        """Load checkpoint and restore curriculum stage."""
        checkpoint_path = os.path.join(self.config["out_dir"], "ckpt.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device_type)

            # Restore curriculum stage before calling parent load
            if 'curriculum_stage_idx' in checkpoint:
                self.current_stage_idx = checkpoint['curriculum_stage_idx']
                self.current_stage = self.curriculum.stages[self.current_stage_idx]

                # Update config for this stage
                assert self.current_stage is not None, "Current stage not set"
                self._apply_stage_config(self.current_stage)
                print(f"Resuming at curriculum stage: {self.current_stage.name}")

        # Now load the rest of the checkpoint
        super().load_checkpoint()
