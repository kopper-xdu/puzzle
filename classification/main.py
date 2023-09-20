import argparse
from trainer_ddp import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-e', '--eval', action='store_true')


def main():
    args = parser.parse_args()

    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
