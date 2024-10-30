# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import tqdm
import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Subset, RandomSampler


from models.default_model import segmentator
from barlowtwins.repos import repo_base as repo


parser = argparse.ArgumentParser(description='Barlow Twins Training')

parser.add_argument('--data_path', default = "total_data" ,
                    type=Path, help='data path for training')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=24, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=1e-3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--subset-resample-ratio', default=0.4, type=float)


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:29501'
    else:
        # single-node distributed training
        print('single-node distributed training')
        args.rank = 0
        args.dist_url = 'tcp://localhost:29502'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # stats_file = open(args.checkpoint_dir / 'stats_1.txt', 'a', buffering=1)
        stats_file = open(args.checkpoint_dir / 'stats_twh.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    
    # initialize dataloader
    dataset = repo.train(df_path = args.data_path, crop_size=[64, 128, 128], return_ds=True)
    sample_ds = Subset(dataset, np.arange(int(len(dataset)*args.subset_resample_ratio)))
    random_sampler = RandomSampler(sample_ds)
    sampler = repo.DistributedSamplerWrapper(random_sampler)
    
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    optimizer = optim.AdamW(parameters, lr=args.learning_rate_weights, weight_decay=args.weight_decay, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate_weights, epochs=args.epochs, steps_per_epoch=len(loader))
    
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
    
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        
        # reload dataloader
        dataset = repo.train(df_path = args.data_path, crop_size=[64, 128, 128], return_ds=True)
        sample_ds = Subset(dataset, np.arange(int(len(dataset)*args.subset_resample_ratio)))
        random_sampler = RandomSampler(sample_ds)
        sampler = repo.DistributedSamplerWrapper(random_sampler)

        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=per_device_batch_size, num_workers=args.workers,
            pin_memory=True, sampler=sampler)
    
        sampler.set_epoch(epoch)
        for step, (y1, y2) in tqdm.tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), desc=f'Epoch {epoch}'):
            
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)

            # adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, loss_stats = model.forward(y1, y2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            
            skip_lr_sched = (scale > scaler.get_scale())
            
            if not skip_lr_sched:
                scheduler.step()
            
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    stats.update(loss_stats)
                    # print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        if args.rank == 0:
            # save checkpoint
            state = dict(
                epoch=epoch + 1, 
                model=model.state_dict(),
                optimizer=optimizer.state_dict(), 
                scheduler=scheduler.state_dict()
            )
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
            torch.save(model.module.backbone.state_dict(), args.checkpoint_dir / 'checkpoint.pth')
    # if args.rank == 0:
    #     # save final model
    #     torch.save(model.module.backbone.state_dict(), args.checkpoint_dir / 'resnet50.pth')


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        segmentation = segmentator()
        decoders = [512, 256, 128, 64, 32]
        projectors = [1408, 640, 256, 128, 64]
        
        self.backbone = segmentation

        # projector
        local_projectors = [nn.Sequential(*[
            nn.Linear(d, p, bias=False),
            nn.BatchNorm1d(p),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(p, p, bias=False),
            nn.BatchNorm1d(p),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(p, p, bias=False),
            nn.BatchNorm1d(p, affine=False),
        ]) for d, p in zip(decoders, projectors)]
        
        self.local_projectors = nn.ModuleList(local_projectors)

    def forward(self, y1, y2):
        n_patch = y1.shape[1]
        y1 = y1.flatten(end_dim=1)[:, None]
        y2 = y2.flatten(end_dim=1)[:, None]
        
        f1s, reconstruct1 = self.backbone(y1)
        f2s, reconstruct2 = self.backbone(y2)

        loss = 0
        for f1, f2, p in zip(f1s, f2s, self.local_projectors):
            
            d, h, w = f1.shape[-3:]
            
            f1 = f1.mean(dim=[-1, -2, -3])
            f2 = f2.mean(dim=[-1, -2, -3])
            
            f1 = p(f1)
            f2 = p(f2)
            
            # empirical cross-correlation matrix
            c = f1.T @ f2

            # sum the cross-correlation matrix between all gpus
            c.div_(self.args.batch_size * n_patch)
            torch.distributed.all_reduce(c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            
            local_loss = on_diag + self.args.lambd * off_diag + (f1 - f2).pow(2).sum(dim=-1).mean()*0.01

            loss = loss + local_loss
        
        recon_loss = F.mse_loss(reconstruct1, y1) + F.mse_loss(reconstruct2, y2)
        loss = loss*0.001
        
        loss_stats = {'local loss': loss.item(), 'recon loss': recon_loss.item()}
        
        loss = loss + recon_loss

        return loss, loss_stats

if __name__ == '__main__':
    main()