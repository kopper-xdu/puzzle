import torch as th
import os
import time
import wandb
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.optim as optim
from datasets.dataset import FFHQ
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from utils import (
    setup_seed,
    setup_DDP,
    get_config,
    setup_logger,
    DataLoaderX,
    cleanup_DDP
)
from models.model import pSp
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import save_image


th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'


class Trainer:
    def __init__(self) -> None:
        pass

    def train(self):
        self.logger = logging.getLogger()
        self.exp_dir = 'experiment/' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
        os.makedirs(self.exp_dir)
        os.system(f"cp ./config.yaml ./{self.exp_dir}")
        logger = logging.getLogger(name='train_logger')
        setup_logger(logger, self.exp_dir + '/train_log.txt')

        self.config = get_config('./config.yaml')
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.config['gpu_ids']
        world_size = self.config['world_size']
        th.multiprocessing.spawn(self.train_loop,
                                 args=(world_size, ),
                                 nprocs=world_size,
                                 join=True)

    def train_loop(self, rank, world_size):
        try:
            setup_DDP(rank, world_size)
            setup_seed(3407 + rank)

            if rank == 0:
                wandb.init(project='t')

            exp_dir = self.exp_dir
            logger = self.logger
            cfg = self.config
            w1 = cfg['w1']
            w2 = cfg['w2']
            w3 = cfg['w3']
            amp = cfg['amp']
            epochs = cfg['epochs']
            optim_cfg = cfg['optim']
            batch_size = cfg['batch_size']
            num_workers = cfg['num_workers']
            log_interval = cfg['log_freq']
            save_interval = cfg['save_epoch']
            log_img_interval = cfg['log_img_epoch']
            logger = logging.getLogger(name='train_logger')

            model = None  # TODO
            model.to(rank)
            # model.load_state_dict(th.load())
            model = FSDP(model, device_ids=[rank], output_device=rank)
            
            # opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **optim_cfg)
            opt = optim.SGD(model.parameters(), **optim_cfg)
            # scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            scheduler = optim.lr_scheduler.MultiStepLR(opt, [100, 200], 0.1)
            scheduler_warmup = GradualWarmupScheduler(
                opt, multiplier=1, total_epoch=3, after_scheduler=scheduler)

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            dataset = None # TODO
            sampler = DistributedSampler(dataset)
            train_loader = DataLoaderX(dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers,
                                      sampler=sampler,
                                      pin_memory=True)
            scaler = GradScaler(enabled=amp)
            
            for epoch in range(epochs):
                for i, img in enumerate(train_loader):
                    B = img.shape[0]
                    img = img.to(rank).float()

                    with autocast(enabled=amp):
                        output = model(img)
                        
                        loss = w1 * id_loss + w2 * per_loss + w3 * kp_loss

                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                    if (i + 1) % log_interval == 0 and rank == 0:
                        loss_item = loss.item()
                        per_loss_item = per_loss.item()
                        id_loss_item = id_loss.item()
                        kp_loss_item = kp_loss.item()
                        print(f'Epoch [{epoch + 1}/ {epochs}], \
                              loss: {loss.item()}, \
                              id_loss: {id_loss_item}, \
                              per_loss: {per_loss_item}, \
                              kp_loss: {kp_loss_item}')
                        # logger.info(f'Epoch [{epoch + 1}/ {epochs}], \
                        #             loss: {loss.item()}, \
                        #             id_loss: {id_loss_item}, \
                        #             l2_loss: {l2_loss_item}, \
                        #             noise_decay_loss: {noise_decay_loss_item}')
                        wandb.log({'loss': loss_item, 
                                   'id_loss': id_loss_item, 
                                   'per_loss': per_loss_item, 
                                   'kp_loss': kp_loss_item})

                scheduler_warmup.step()

                if (epoch + 1) % save_interval == 0 and rank == 0:
                    th.save(model.module.state_dict(), exp_dir +
                            f'/ckpt-epoch{epoch + 1}.pth')
                    logger.info('checkpoint saved!')
                if (epoch + 1) % log_img_interval == 0 and rank == 0:
                    # log img
                    pass

            if rank == 0:
                th.save(model.module.state_dict(),
                        exp_dir + f'/ckpt-final.pth')
                logger.info('model saved!')

        except Exception as ex:
            print(ex)
            
        cleanup_DDP()
            
    @th.no_grad()
    def test(self, ckpt_path):
        cfg = get_config('config.yaml')
        init_ckpt_path = cfg['init_ckpt_path']

        model = pSp()
        model.to(0)
        model.load_init_weights(init_ckpt_path)
        model.load_state_dict(th.load(ckpt_path))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = FFHQ(
            train=True, data_dir='./FFHQ', transform=transform)
        loader = DataLoaderX(dataset,
                            batch_size=4,
                            num_workers=0
                                  )
        loader = iter(loader)
        x = loader.next().cuda()
        inv_imgs, latents = model.restyle(x)
        attack_imgs, _, _ = model(latents, is_latent=True, attack=True)

        save_image((inv_imgs + 1) / 2, './test/inversion.png')
        save_image((x + 1) / 2, './test/origin.png')
        save_image((attack_imgs + 1) / 2, './test/attack.png')


    @th.no_grad()
    def eval(self, exp_dir):
        cfg = get_config(os.path.join(exp_dir, 'config.yaml'))
        ckpt_path = cfg['ckpt_path']

        model = pSp()
        model.to(0)
        model.load_state_dict(th.load(ckpt_path))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        dataset = FFHQ(
            train=False, data_dir='./FFHQ', transform=transform)
        test_loader = DataLoaderX(dataset,
                                  batch_size=16,
                                  num_workers=2
                                  )

        for img in test_loader:
            B = img.shape[0]
            img = img.cuda()
            out = model(img)



if __name__ == '__main__':
    trainer = Trainer()
    trainer.test('experiment/20230228-21-52-57/ckpt-epoch10.pth')