"""
Wrapper script to launch SFT.
"""

if __name__ == "__main__":
    import argparse
    from training.sft.train_sft import SFTTrainer

    parser = argparse.ArgumentParser(description="Fine-tune GPT model with SFT")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    args = parser.parse_args()

    trainer = SFTTrainer(resume=args.resume)
    trainer.train()
