import time
import os
from os.path import join
import argparse

import logging
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore import nn, ops
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model

from training.losses import CustomLossCell

from training.dataset import *
from training.network import define_network

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
                         'fastest way to use multiprocessing for either single node or '
                         'multi node data parallel training')
# +---------------------------------------------------------+

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

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
    

    if args.multiprocessing_distributed:
        print('multiprocessing_distributed')
        init()
        ngpus_per_node = mindspore.communication.get_group_size()
        args.world_size = ngpus_per_node * args.world_size
        mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.HYBRID_PARALLEL)
    else:
        print('No Distributed Data Processing. GPU => ', args.gpu)
        main_worker(args.gpu, 1, args)
    
def main_worker(gpu, ngpus_per_node, args):
    global best_auc1
    if args.local_rank == 0:
        logging.basicConfig(filename=join(args.results, 'train.log'), filemode='w', format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    args.gpu = gpu
    if args.gpu is not None:
        print('Use GPU: {} for training.'.format(args.gpu))
    if args.distributed:
        print("distributed == True")
        if args.dist_url == "env://" and args.local_rank == -1:
            args.local_rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.local_rank = args.local_rank * ngpus_per_node + gpu
        mindspore.communication.init(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)


    if args.distributed:
        if args.gpu is not None:
            args.batch_size = int(config.batch_size / ngpus_per_node)
            args.workers = int((config.workers + ngpus_per_node -1) / ngpus_per_node)
            mindspore.set_context(args.gpu)
            print('GPU:', args.gpu)

        else:
            '''
            DistributedDataParallel will divide and allocate batch_size to all
            available GPUs if device_ids are not set
            '''
            mindspore.set_auto_parallel_context(dataset_strategy="full_batch")
            print('Allocate batch-size to all available GPUs')

    elif args.gpu is not None:
        # Single GPU.
        mindspore.set_context(device_id=args.gpu, device_target="GPU")
        
        print("Use Single GPU", args.gpu)
    else:
        print(">>> Use CPU")
        mindspore.set_context(device_target="CPU")

    model = define_network(f_dim=512, h_didm=128, a_dim=12, config=config)

    if args.resume:
        if args.local_rank % ngpus_per_node == 0:
            print('=> Use previously trained model from {}'.format(args.resume))
        
        config.start_epoch, best_auc1 = load_model(model, join(args.resume, f'{config.best_ckpt}'))
    opt = mindspore.nn.AdamWeightDecay(model.trainable_params(), learning_rate=config.lr)
    milestone = [item for item in range(0, config.epochs, config.scheduler_step)]
    learning_rates = []
    init_gamma = 1
    for _ in range(config.epochs // config.scheduler_step):
        learning_rates.append(init_gamma)
        init_gamma *= 0.5
    scheduler = mindspore.nn.piecewise_constant_lr(milestone[1:], learning_rates[1:])
    
    criterion = mindspore.nn.CrossEntropyLoss()

    if config.train_dataset['datalabel'] == 'celeb':
        dataset_generator = DFDataset(phase='train', tag="", codec='', **config.train_dataset)
        dataset = ds.GeneratorDataset(dataset_generator, ["img", "label"], shuffle=False)
        train_data, val_data = dataset.split([0.8, 0.2])

    else:
        train_data = DFDataset(phase='train', tag="", codec='c23', **config.train_dataset)
        val_data   = DFDataset(phase='val', datalabel="ff-all", tag="", codec='c23', **config.val_dataset)

    if args.distributed:
        train_sampler = ds.DistributedSampler(train_data, shuffle=True)
        val_sampler = ds.DistributedSampler(val_data, shuffle=True)
    else:
        train_sampler = val_sampler = None

    train_dataloader = ds.GeneratorDataset(train_data, ["img", "label"], shuffle=True, sampler=train_sampler)
    train_dataloader = train_dataloader.batch(batch_size=config.batch_size)
    train_dataloader = train_dataloader.map(operations=mindspore.dataset.transforms.TypeCast(mindspore.int32), input_columns="label")

    val_dataloader = ds.GeneratorDataset(val_data, ["img", "label"], shuffle=(val_sampler is None), sampler=val_sampler)
    val_dataloader = val_dataloader.batch(batch_size=4)
    val_dataloader = val_dataloader.map(operations=mindspore.dataset.transforms.TypeCast(mindspore.int32), input_columns="label")
    # model = Model(model, criterion, opt)
    
    for epoch in range(config.start_epoch, config.epochs):
        train(train_dataloader, model, opt, criterion, epoch, args)
        val_auc = valid(val_dataloader, model, criterion, ngpus_per_node, epoch, args)
        scheduler.step()

        is_best = val_auc > best_auc1
        best_auc1 = max(val_auc, best_auc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.local_rank % ngpus_per_node==0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc1': best_auc1
                }, 
            is_best, args.checkpoint, filename=config.ckpt, best=config.best_ckpt)
        
def train(train_dataloader, model, optimizer, criterion, epoch, args):
    # loss_net = CustomLossCell(model, criterion)
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss
    
    model.set_train()
    size = train_dataloader.get_dataset_size()
    for index, (data, label) in enumerate(train_dataloader.create_tuple_iterator()):
        loss = train_step(data, label)
        if index % 100 == 0:
            loss, current = loss.asnumpy(), index
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
    return loss

def valid(val_dataloader, model, criterion, ngpus_per_node, epoch, args):
    model = Model(model, loss_fn=criterion, optimizer=None, metrics={'acc'})
    acc = model.eval(val_dataloader, dataset_sink_mode=False)

    return acc

if __name__ == '__main__':
    main()
