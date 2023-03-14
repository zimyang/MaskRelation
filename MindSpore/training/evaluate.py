import mindspore
from mindspore import ops
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def compute_video_level_auc(video_to_logits, video_to_labels):
    """ "
    Compute video-level area under ROC curve. Averages the logits across the video for non-overlapping clips.

    Parameters
    ----------
    video_to_logits : dict
        Maps video ids to list of logit values
    video_to_labels : dict
        Maps video ids to label
    """
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])

    fpr, tpr, _ = metrics.roc_curve(output_labels.cpu().numpy(), output_batch.cpu().numpy())
    return metrics.auc(fpr, tpr)

def compute_video_level_acc(video_to_logits, video_to_labels):
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    prediction = (output_batch>=0.5).long()
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])
    acc = metrics.accuracy_score(output_labels.cpu().numpy(), prediction.cpu().numpy())
    return acc

def compute_video_level_prf(video_to_logits, video_to_labels):
    output_batch = torch.stack(
        [torch.mean(torch.stack(video_to_logits[video_id]), 0, keepdim=False) for video_id in video_to_logits.keys()]
    )
    prediction = (output_batch>=0.5).long()
    output_labels = torch.stack([video_to_labels[video_id] for video_id in video_to_logits.keys()])
    pre, rec, f1, support = metrics.precision_recall_fscore_support(output_labels.cpu().numpy(), prediction.cpu().numpy(), average='binary')
    return pre, rec, f1, support