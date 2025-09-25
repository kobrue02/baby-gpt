"""
Wrapper script to launch SFT.
"""

if __name__ == "__main__":
    from training.sft.train_sft import SFTTrainer

    trainer = SFTTrainer()
    trainer.train()
