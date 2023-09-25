import os, sys
import traceback
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

import torch as th
import torchvision
import torch.optim as optim
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from models import MyViT
from datasets import dataset
from utils import (
    setup_seed,
    setup_ddp,
    cleanup_ddp,
    init_exp,
    DataLoaderX
)


class Trainer:
    def __init__(self, config) -> None:
        self.config = config

        cuda_config = config.cuda_config
        th.backends.cudnn.deterministic = cuda_config.cudnn_deterministic
        th.backends.cudnn.benchmark = cuda_config.cudnn_benchmark
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_config.cuda_visible_devices

    def train(self):
        self.exp_dir = init_exp(self.config)
        world_size = self.config.cuda_config.world_size

        th.multiprocessing.spawn(self.train_loop,
                                 args=(world_size, ),
                                 nprocs=world_size,
                                 join=True)

    def train_loop(self, rank, world_size):
        try:
            setup_ddp(rank, world_size, self.config.cuda_config.port)
            setup_seed(3407 + rank)

            exp_dir = self.exp_dir
            config = self.config

            if rank == 0:
                wandb.init(project='classify',
                           name=exp_dir[11:],
                           config=OmegaConf.to_container(config)
                           )
                
            model = MyViT(mode=config.mode, **config.model_param)
            model.to(rank)
            if config.ckpt_path:
                model.load_state_dict(th.load(config.ckpt_path), strict=False)
            model = DDP(model, device_ids=[rank], output_device=rank)
            
            opt = getattr(optim, config.optim.optim_name)(filter(lambda p: p.requires_grad, model.parameters()),
                                                          **config.optim.optim_param)
            scheduler = getattr(optim.lr_scheduler, config.scheduler.scheduler_name)(opt, **config.scheduler.scheduler_param)
            
            if config.use_warmup:
                warmup = GradualWarmupScheduler(opt, **config.warmup_param, after_scheduler=scheduler)
                opt.zero_grad()
                opt.step()
                warmup.step()

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            dataset = getattr(torchvision.datasets, config.dataset)(root=f'./datasets/{config.dataset}', 
                                                                    transform=transform, 
                                                                    download=True)
            sampler = th.utils.data.distributed.DistributedSampler(dataset)
            train_loader = DataLoaderX(dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      sampler=sampler,
                                      pin_memory=True,
                                      persistent_workers=True if config.num_workers > 0 else False
                                      )
            
            scaler = GradScaler(enabled=config.amp)
            loss_fn = th.nn.CrossEntropyLoss()
            
            for epoch in range(config.epochs):
                for i, (img, tgt) in enumerate(train_loader):
                    img = img.to(rank)
                    tgt = tgt.to(rank)

                    with autocast(enabled=config.amp, cache_enabled=False):
                        out = model.forward(img)
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
                    self.eval(ckpt_path, use_wandb=True)

            if rank == 0:
                ckpt_path = os.path.join(exp_dir, 'ckpt-final.pth')
                th.save(model.module.state_dict(), ckpt_path)
                print.info('model saved!')

                self.eval(ckpt_path, use_wandb=True)
            
            if rank == 0:
                wandb.finish()

        except Exception as ex:
            error_type, error_value, error_trace = sys.exc_info()
            for info in traceback.extract_tb(error_trace):  
                print(info)
            print(error_value)

    @th.no_grad()
    def eval(self, ckpt_path, use_wandb = False):
        config = self.config

        model = MyViT(mode=config.mode, **config.model_param)
        model.to(0)
        model.load_state_dict(th.load(ckpt_path))
        model.eval()

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        dataset = getattr(torchvision.datasets, config.dataset)(root=f'./datasets/{config.dataset}',
                                                                train=False,
                                                                transform=transform, 
                                                                download=True
                                                                )

        test_loader = DataLoaderX(dataset, 
                                  batch_size=256,
                                  num_workers=2,
                                  )

        total = 0
        correct = 0
        loss = 0
        loss_fn = th.nn.CrossEntropyLoss()

        for img, tgt in tqdm(test_loader):
            img = img.to(0)
            tgt = tgt.to(0)

            out = model(img)

            loss += loss_fn(out, tgt).item()

            _, pred = th.max(out, dim=1)
            correct += sum(pred == tgt)
            total += pred.shape[0]

        acc = correct / total
        # loss /= len(test_loader)
        print(f'eval_loss: {loss}, eval_acc: {acc}')

        if use_wandb:
            wandb.log({'eval_loss': loss,
                       'eval_acc': acc})
