import time
import os
from os.path import join
import argparse

import logging
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from training.dataset import *
from training.network import define_network
from training.evaluate import *
from utils.util import *
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoint/')
parser.add_argument('--results', type=str, default='results/')
parser.add_argument('--resume', type=str, default=None)

parser.add_argument('--lambda_bce', type=float, default=1., help="1")

parser.add_argument('--print_iter', type=int, default=100, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)

parser.add_argument('--amp', action='store_true', help='if True, use fp16.')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist_url',   type=str, default='tcp://127.0.0.1:23504')
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--gpu',        type=int, default=None, help='if DDP, set None.')
parser.add_argument('--multiprocessing_distributed', type=bool, default=False,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# +---------------------------------------------------------+

def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = True

# +---------------------------------------------------------+
"""
Setup Configuration
"""
config = Config(datalabel='ff-all', recipes=['ff-all','r3d'], ckpt='CKPT_FILE_NAME.pth.tar', best_ckpt='BEST_CKPT_FILE_NAME.pth.tar', epochs=100)
best_auc1 = 0

def main():
    args = parser.parse_args()
    setup_seed(1000)
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    if args.multiprocessing_distributed:
        print("args.multiprocessing_distributed==True")
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print('No Distributed Data Processing. GPU => ', args.gpu)
        main_worker(args.gpu, ngpus_per_node, args)
    
def main_worker(gpu, ngpus_per_node, args):
    global best_auc1
    if args.local_rank == 0:
        logging.basicConfig(filename=join(args.results, 'train.log'), filemode='w', format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    args.gpu = gpu
    if args.gpu is not None:
        print('Use GPU: {} for training.'.format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)

    model = define_network(f_dim=512, h_didm=128, a_dim=12, config=config)

    if args.resume:
        if args.local_rank % ngpus_per_node == 0:
            print('=> Use previously trained model from {}'.format(args.resume))
        
        config.start_epoch, best_auc1 = load_model(model, join(args.resume, f'{config.best_ckpt}'))

    if args.distributed:
        if args.gpu is not None:
            args.batch_size = int(config.batch_size / ngpus_per_node)
            args.workers = int((config.workers + ngpus_per_node -1) / ngpus_per_node)
            torch.cuda.set_device(args.gpu)
            print('GPU:', args.gpu)
            
            model = model.cuda(args.gpu)
            model = DDP(model, device_ids=[args.gpu])
        else:
            '''
            DistributedDataParallel will divide and allocate batch_size to all
            available GPUs if device_ids are not set
            '''
            model = model.cuda()
            
            model = DDP(model)
            print('Allocate batch-size to all available GPUs')

    elif args.gpu is not None:
        # Single GPU.
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print("Use Single GPU", args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        print("Use Data Parallel.")

    opt = optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    if config.train_dataset['datalabel'] == 'celeb':
        dataset = DFDataset(phase='train', tag="", codec='', **config.train_dataset)
        len_dataset = len(dataset)
        train_size = int(0.8 * len_dataset)
        val_size = len_dataset - train_size
        train_data, val_data = torch.utils.data.random_split(
            dataset=dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(1000)
        )
    else:
        train_data = DFDataset(phase='train', tag="", codec='c23', **config.train_dataset)
        val_data = DFDataset(phase='val', datalabel="ff-all", tag="", codec="c23", **config.val_dataset)

    if args.distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=True)
    else:
        train_sampler = val_sampler = None

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=(train_sampler is None), num_workers=config.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=(val_sampler is None), num_workers=config.workers, pin_memory=True, drop_last=True, sampler=val_sampler)

    for epoch in range(config.start_epoch, config.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train(train_dataloader, model, opt, criterion, scaler, epoch, args)
        val_auc = valid(val_dataloader, model, criterion, ngpus_per_node, epoch, args)
        scheduler.step()

        is_best = val_auc > best_auc1
        best_auc1 = max(val_auc, best_auc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.local_rank % ngpus_per_node==0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc1': best_auc1
            }, is_best, args.checkpoint, filename=config.ckpt, best=config.best_ckpt)
        
def train(train_dataloader, model, optimizer, criterion, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':1.2f')
    
    loss_ce = AverageMeter('CE', ':.3f')
    loss_time = AverageMeter('Loss_tc', ':.3f')
    loss_org = AverageMeter('Loss_oc', ':.3f')
    loss_reg = AverageMeter('Reg', ':.3f')
    acc = AverageMeter('Acc', ':3.2f')

    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time, data_time, loss_ce, loss_time, loss_org, loss_reg, acc],
        prefix="Epoch:{}".format(epoch)
    )
    
    end = time.time()
    model.train()

    for idx, data in enumerate(train_dataloader):

        data_time.update(time.time() - end)

        imgs, labels = data['img'], data['label']
        _batch_size = labels.shape[0]

        if args.amp:
            imgs = imgs.half()
        else:
            imgs = imgs.float()
        
        labels = labels.long()
        
        if torch.cuda.is_available():
            imgs = imgs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
        
        ## Deepfakes Detection branch
        optimizer.zero_grad()
        with autocast(args.amp):
            y_hat, loss_tc, loss_oc = model(imgs)
            
            reg = model.l2_reg()

            loss_bce = args.lambda_bce * criterion(y_hat, labels)
            
            loss = loss_bce + reg + 0.75*loss_tc + 0.75*loss_oc

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1 = accuracy(y_hat, labels)
        
        acc.update(acc1[0].item(), _batch_size)
        loss_ce.update(loss_bce.item(), _batch_size)
        loss_time.update(loss_tc.item(), _batch_size)
        loss_org.update(loss_oc.item(), _batch_size)
        loss_reg.update(reg.item(), _batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_iter == 0 and args.local_rank == 0 :
            progress.display(idx)


def valid(val_dataloader, model, criterion, ngpus_per_node, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':1.2f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeter('Acc', ':3.2f')
    
    progress = ProgressMeter(
        len(val_dataloader),
        [batch_time, data_time, losses, acc],
        prefix='Test:')
    
    # switch to evaluate mode
    model.eval()

    nplabels = []
    softmax_logits = []
    int_logits = []

    with torch.no_grad():
        end = time.time()

        for idx, data in enumerate(val_dataloader):
            img, label = data['img'], data['label']

            _batch_size = label.shape[0]

            data_time.update(time.time() - end)
            
            img = img.float() 
            label = label.long()
            
            if torch.cuda.is_available():
                img = img.cuda(args.gpu, non_blocking=True)
                label = label.cuda(args.gpu, non_blocking=True)
            
            logits = model(img)
            
            loss_ce = args.lambda_bce * criterion(logits, label)
            loss = loss_ce

            acc1 = accuracy(logits, label)

            nplabels.append(label.cpu().numpy().astype(np.int8))
            int_logits.append(logits.max(1)[1].cpu().numpy().astype(np.int8))
            softmax_logits.append(logits.softmax(1)[:,1].detach().cpu().numpy())

            losses.update(loss.item(), _batch_size)
            acc.update(acc1[0].item(), _batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_losses = loss.clone().detach().cuda(args.gpu)
            epoch_acc = torch.tensor(acc1).cuda(args.gpu)
            
            if args.distributed:
                dist.all_reduce(epoch_losses, op=dist.ReduceOp.SUM)
                dist.all_reduce(epoch_acc, op=dist.ReduceOp.SUM)

            epoch_losses = epoch_losses.item() / ngpus_per_node
            epoch_acc = epoch_acc.cpu().numpy() / ngpus_per_node

            batch_info = 'Loss:{:.4f} Acc:{:.2f}'.format(epoch_losses, acc.avg)

            if idx % args.print_iter == 0 and args.local_rank == 0:
                progress.display(idx)
    
    y_true = np.concatenate(nplabels)
    y_pred = np.concatenate(softmax_logits)
    int_pred = np.concatenate(int_logits)

    auc = roc_auc_score(y_true, y_pred)
    auc = 100. * auc
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, int_pred, average='binary')
    pre = 100.* pre; rec = 100. * rec; f1 = 100. * f1


    if args.local_rank % ngpus_per_node == 0:
        logging.info('Train Epoch:{} Time: {} {}'.format(epoch, batch_time, batch_info))
  
        print(" * Acc:{Acc.avg:.3f} Auc:{Auc:.3f} Pre:{Pre:.3f} Rec:{Rec:.3f} F1:{F1:.3f}".format(Acc=acc, Auc=auc, Pre=pre, Rec=rec, F1=f1))

    return acc.avg

if __name__ == '__main__':
    main()
