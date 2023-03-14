from os.path import join
import mindspore
import shutil

# import matplotlib
# matplotlib.use('agg')


def set_lr(opt, new_lr):
    for param_group in opt.param_groups:
        param_group["lr"] = new_lr

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', best='best.pth.tar'):
    mindspore.save_checkpoint(state, join(checkpoint, filename))
    if is_best:
        shutil.copyfile(join(checkpoint, filename), join(checkpoint, best))

def load_model(model, pretrained):
    weights = mindspore.load_checkpoint(pretrained)
    epoch = weights['epoch']
    best_auc1 = weights['best_auc1']
    pretrained_dict = weights["state_dict"]
    model_dict = model.parameters_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    mindspore.load_param_into_net(model, model_dict)
    del weights
    return epoch, best_auc1

def pretrained_model(model, pretrained):
    pretrained_dict = mindspore.load_checkpoint(pretrained)
    model_dict = model.parameters_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    mindspore.load_param_into_net(model, model_dict)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
 

## model parameters
def summary(model):
    total_params = sum(p.numel() for p in model.get_parameters())
    trainable_params = sum(p.numel() for p in model.get_parameters() if p.requires_grad)
    print('Total - %.2fM' % (total_params/1e6))
    print('Trainable - %.2fM' % (trainable_params/1e6))
