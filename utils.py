import torch
import numpy as np
import random
import os
import time
import platform
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_ddp(rank, world_size, port='12001'):
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = 'localhost'
    torch.cuda.set_device(rank)

    plat = platform.system().lower()
    backend = 'nccl' if plat == 'linux' else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def init_exp(config):
    exp_dir = os.path.join('experiment', config.exp_name)
    os.makedirs(exp_dir)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    return exp_dir


# def setup_logger(logger, log_file, level=logging.DEBUG):
#     logger.setLevel(level)

#     fh = logging.FileHandler(log_file, mode='a')
#     fh.setLevel(level)

#     ch = logging.StreamHandler()
#     ch.setLevel(level)

#     formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)

#     logger.addHandler(fh)
#     logger.addHandler(ch)

#     return logger


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
