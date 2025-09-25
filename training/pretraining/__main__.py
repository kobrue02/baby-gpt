"""
Wrapper script to launch training.
"""

if __name__ == "__main__":
    from training.pretraining.training_utils import PreTrainer

    trainer = PreTrainer()
    trainer.train()
