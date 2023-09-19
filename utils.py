import torch
import numpy as np
import random
import os
import time
import torch.distributed as dist
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_ddp(rank, world_size):
    os.environ['MASTER_PORT'] = '12001'
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.system('export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)')
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def init_exp(config_path):
    exp_dir = os.path.join('experiment', time.strftime("%Y%m%d-%H-%M-%S", time.localtime()))
    os.makedirs(exp_dir)
    os.system(f"cp {config_path} ./{exp_dir}")

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
