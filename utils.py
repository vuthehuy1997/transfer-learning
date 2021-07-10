import os
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def get_label(dir_path):
    label_dir = os.path.basename(os.path.dirname(dir_path))
    if label_dir == '0':
        return 'binh_thuong'
    elif label_dir == '1':
        return '16+'
    elif label_dir == '2':
        return '18+'
    else:
        return 'error'

def get_data(root_dir):
    list_of_files = []
    for (dirpath, dirnames, filenames) in tqdm(os.walk(root_dir)):
        for filename in filenames:
            if '.jpg' in filename:
                list_of_files.append(os.path.join(dirpath, filename))
    images_dir_paths = list(set([f[:f.rfind('/')] for f in list_of_files]))
    images_dir_paths = [x for x in images_dir_paths if "-1" not in x.split('/')]
    labels = [get_label(dir_path) for dir_path in images_dir_paths]
    
    assert (len(images_dir_paths)==len(labels)), "Number of videos and labels must be equal"
    return images_dir_paths, labels


def get_data2train(images_root_dir, data_labels, subset):
    """
        images_root_dir:
            |__ 18+ ... (30 .JPG(s))
            |__ 16+
            |__ binh_thuong
            |__ update
                |__ 18+
                |__ 16+
                |__ binh_thuong
        Returns:
            + paths
            + labels
    """
    total_dir_paths, total_labels = [], []
    if isinstance(images_root_dir, list):
        for root_dir in images_root_dir:
            dir_paths, labels = get_data(root_dir)
            total_dir_paths.extend(dir_paths)
            total_labels.extend(labels)
    else:
        total_dir_paths, total_labels = get_data(images_root_dir)
    print('Total: ', Counter(total_labels))
    if subset:
        random.seed(0)
        idxes = []
        for data_label in data_labels:
            idxes.extend(random.choices([i for i, x in enumerate(total_labels) if x == data_label], k=5000))
        total_dir_paths = [total_dir_paths[idx] for idx in idxes]
        total_labels = [total_labels[idx] for idx in idxes]
    return total_dir_paths, total_labels
        
def get_data2test(test_dir):
    data_labels = ['18+', '16+', 'binh_thuong']
    ALLOWED_EXTENSIONS = ('mp4', 'm4a', 'm4v', 'f4v', 'f4a', 'm4b', 'm4r', 'f4b', 'mov', '3gp', '3gp2', '3g2', '3gpp', '3gpp2', 'ogg', 'oga', 'ogv', 'ogx', 'wmv', 'wma', 'avi', 'mpg', 'flv', 'webm', 'mkv', 'ts')
    video_paths = []
    for (dirpath, dirnames, filenames) in tqdm(os.walk(test_dir)):
        for filename in filenames:
            if filename.lower().endswith(ALLOWED_EXTENSIONS):
                video_paths.append(os.path.join(dirpath, filename))

    labels = []
    for video_path in video_paths:
        for label in data_labels:
            if label in video_path:
                labels.append(label)
                break
    assert (len(video_paths)==len(labels)), "Number of videos and labels must be equal"
    print("Num test videos: ", len(video_paths))
    return video_paths, labels

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, target, logits):
        '''
        Shapes:
        -------
        logits: [N, num_classes]
        target: [N]
        '''
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1))
        prob = log_prob.detach().exp()

        if self.alpha is not None:
            alpha = self.alpha.to(target.device)
            alpha = alpha.expand(len(target), logits.size(-1)).detach().float()
            log_prob = (alpha * log_prob)
        
        loss = -(1 - prob)**self.gamma * log_prob

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            if self.alpha is None:
                return loss.mean()
            else:
                return loss.sum() / alpha.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('Invalid reduction')

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
