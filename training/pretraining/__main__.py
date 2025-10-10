"""
Wrapper script to launch training.
"""

if __name__ == "__main__":
    import argparse
    from training.pretraining.pretrainer import PreTrainer

    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    args = parser.parse_args()

    trainer = PreTrainer(resume=args.resume)
    trainer.train()
