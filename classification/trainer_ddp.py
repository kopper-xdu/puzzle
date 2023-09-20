import os, sys
import traceback
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from utils import (
    setup_seed,
    setup_ddp,
    init_exp,
    DataLoaderX
)

import torch as th
import torchvision
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from warmup_scheduler import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from models.model import ViT
from datasets import dataset


class Trainer:
    def __init__(self, config_path = 'config.yaml') -> None:
        self.config_path = config_path
        self.conf = OmegaConf.load(config_path)

        cuda_conf = self.conf.cuda_config
        th.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
        th.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_conf.cuda_visible_devices

    def train(self):
        self.exp_dir = init_exp(self.config_path)
        world_size = self.conf.cuda_config.world_size

        th.multiprocessing.spawn(self.train_loop,
                                 args=(world_size, ),
                                 nprocs=world_size,
                                 join=True)

    def train_loop(self, rank, world_size):
        try:
            setup_ddp(rank, world_size)
            setup_seed(3407 + rank)

            exp_dir = self.exp_dir
            config = self.conf

            if rank == 0:
                wandb.init(project='classify',
                           name=exp_dir,
                           config=OmegaConf.to_container(config)
                           )
            
            model = ViT(**config.model_param)
            model.to(rank)
            model = DDP(model, device_ids=[rank], output_device=rank)
            
            opt = getattr(optim, config.optim.optim_name)(filter(lambda p: p.requires_grad, model.parameters()),
                                                          **config.optim.optim_param)
            scheduler = getattr(optim.lr_scheduler, config.scheduler.scheduler_name)(opt, **config.scheduler.scheduler_param)
            if config.use_warmup:
                warmup = GradualWarmupScheduler(opt, **config.warmup_param, after_scheduler=scheduler)

            transform = transforms.Compose([
                # transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', transform=transform, download=True)
            sampler = th.utils.data.distributed.DistributedSampler(dataset)
            train_loader = DataLoaderX(dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      sampler=sampler,
                                      pin_memory=True)
            
            scaler = GradScaler(enabled=config.amp)
            loss_fn = th.nn.CrossEntropyLoss()

            for epoch in range(config.epochs):
                # for i, (img, tgt) in enumerate(tqdm(train_loader)):
                for i, (img, tgt) in enumerate(train_loader):
                    img = img.to(rank)
                    tgt = tgt.to(rank)

                    with autocast(enabled=config.amp):
                        out = model(img)
                        loss = loss_fn(out, tgt)

                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    if (i + 1) % config.log_freq == 0 and rank == 0:
                        print(f'Epoch [{epoch + 1}/ {config.epochs}], loss: {loss.item()}')
                        wandb.log({'loss': loss.item(),
                                   'lr': opt.state_dict()['param_groups'][0]['lr']})

                if config.use_warmup:
                    warmup.step()
                else:
                    scheduler.step()

                if (epoch + 1) % config.save_epoch == 0 and rank == 0:
                    ckpt_path = os.path.join(exp_dir, f'ckpt-epoch{epoch + 1}.pth')
                    th.save(model.module.state_dict(), ckpt_path)
                    print('checkpoint saved!')

                if (epoch + 1) % config.eval_epoch == 0 and rank == 0:
                    self.eval(ckpt_path, wandb=True)

            if rank == 0:
                ckpt_path = exp_dir + f'/ckpt-final.pth'
                th.save(model.module.state_dict(), ckpt_path)
                print.info('model saved!')

                self.eval(ckpt_path, wandb=True)

            if rank == 0:
                wandb.finish()

        except Exception as ex:
            error_type, error_value, error_trace = sys.exc_info()
            for info in traceback.extract_tb(error_trace):  
                print(info)
            print(error_value)

    @th.no_grad()
    def eval(self, ckpt_path, wandb = False):
        config = self.conf

        model = ViT(**config.model_param)
        model.to(0)
        model.load_state_dict(th.load(ckpt_path))
        model.eval()

        transform = transforms.Compose([
                # transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False, transform=transform, download=True)

        test_loader = DataLoaderX(dataset, 
                                  batch_size=config.batch_size, 
                                  num_workers=config.num_workers
                                  )
        loss_fn = th.nn.CrossEntropyLoss()

        total = 0
        correct = 0
        loss = 0
        for img, tgt in tqdm(test_loader):
            img = img.to(0)
            tgt = tgt.to(0)

            out = model(img)

            loss += loss_fn(out, tgt).item()
            _, pred = th.max(out, 1)
            total += img.shape[0]
            correct += sum(pred == tgt)

        print(f'eval_loss: {loss}')
        wandb.log({'eval_loss': loss,
                   'acc': correct / total})



if __name__ == '__main__':
    trainer = Trainer()
    trainer.test('experiment/20230228-21-52-57/ckpt-epoch10.pth')