"""
Wrapper script to launch training.
"""

from training.training_utils import Trainer


def main():
    trainer = Trainer()
    trainer.train(data_dir='data')


if __name__ == "__main__":
    main()