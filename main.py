import sys
import argparse
import pretrain
import train
from omegaconf import OmegaConf

# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', default='pretrain', type=str)


def main():
    # args = parser.parse_args()

    cli_config = OmegaConf.from_cli(sys.argv[1:])

    if cli_config.mode == 'pretrain':
        config = OmegaConf.load('config.yaml')
    elif cli_config.mode == 'finetune':
        config = OmegaConf.load('ft_config.yaml')
    elif cli_config.mode == 'full':
        config = OmegaConf.load('full_config.yaml')

    config = OmegaConf.merge(config, cli_config)

    if config.mode == 'pretrain':
        trainer = pretrain.Trainer(config)
        trainer.train()
    elif config.mode == 'finetune' or config.mode == 'full':
        trainer = train.Trainer(config)
        trainer.train()


if __name__ == '__main__':
    main()
