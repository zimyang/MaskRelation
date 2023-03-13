import time
import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader

from training.dataset import *
from training.network import define_network
from training.evaluate import *
from utils.util import *
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=str, default='results/')
parser.add_argument('--resume', type=str, default='')

parser.add_argument('--print_iter', type=int, default=50, help='print frequency')
parser.add_argument('--gpu',        type=int, default=None, help='if DDP, set None.')
parser.add_argument('--amp', action='store_true', help='if True, use fp16.')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# +---------------------------------------------------------+
"""
Setup Configuration
"""
config = Config(datalabel="ff-all", recipes=['ff-all', 'r3d'])

def main():
    args = parser.parse_args()
    setup_seed(25)
    os.makedirs(args.results, exist_ok=True)

    model = define_network(f_dim=512, h_didm=128, a_dim=12, config=config)
    
    if args.resume:
        weights = torch.load(args.resume)
        pretrained_dict = weights['state_dict'] 
        model.load_state_dict(pretrained_dict)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print("Use Single GPU", args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        print("Use Data Parallel.")
    
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    val_data = DFDataset(phase='test', datalabel="ff", tag="deepfakes", codec="c23", **config.val_dataset)
    ## deepfakes; face2face; faceswap; neural_textures
    val_dataloader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=config.workers, pin_memory=True, drop_last=True)

    auc1 = valid(val_dataloader, model, criterion, 1, args)

def valid(val_dataloader, model, criterion, ngpus_per_node, args):
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

            _batch_size = img.shape[0]
            data_time.update(time.time() - end)
            
            if args.amp:
                img = img.half() # fp16
            else:
                img = img.float() # fp32
            label = label.long()
            
            if torch.cuda.is_available():
                img = img.cuda(args.gpu, non_blocking=True)
                label = label.cuda(args.gpu, non_blocking=True)
            
            logits = model(img)

            loss_ce = criterion(logits, label)
            loss = loss_ce
            
            pred = torch.nn.functional.softmax(logits, dim=1)
            acc1 = accuracy(pred, label)

            softmax_logits.append(logits.softmax(1)[:,1].detach().cpu().numpy())
            nplabels.append(label.cpu().numpy().astype(np.int8))
            int_logits.append(logits.max(1)[1].cpu().numpy().astype(np.int8))
            
            losses.update(loss.item(), _batch_size)
            acc.update(acc1[0].item(), _batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            epoch_losses = loss.clone().detach().cuda(args.gpu)
            epoch_acc = torch.tensor(acc1).cuda(args.gpu)

            epoch_losses = epoch_losses.item() / ngpus_per_node
            epoch_acc = epoch_acc.cpu().numpy() / ngpus_per_node

            progress.display(idx)
    
    y_true = np.concatenate(nplabels)
    y_pred = np.concatenate(softmax_logits)
    int_pred = np.concatenate(int_logits)
    
    auc = roc_auc_score(y_true, y_pred)
    auc = 100. * auc

    eer1, _ = calculate_eer(y_true, y_pred)
    
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, int_pred, average='binary')
    pre = 100.* pre; rec = 100. * rec; f1 = 100. * f1

    print(" * Acc:{Acc.avg:.3f} Auc:{Auc:.3f} Pre:{Pre:.3f} Rec:{Rec:.3f} F1:{F1:.3f} EER:{Eer:.3f}".format(Acc=acc, Auc=auc, Pre=pre, Rec=rec, F1=f1, Eer=eer1))

    return auc

if __name__ == '__main__':
    main()
