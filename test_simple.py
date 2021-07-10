import os
import time
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import subprocess
import ffmpeg
import datetime
from dataset import MaskDetectionDataset
from torch.utils.data import DataLoader
from model.cnn import CNNModel
# from model.cnn_simple import CNNModel
import math
from sklearn.metrics import confusion_matrix
from utils import AverageMeter
import yaml
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

def evaluate(weight, config_path):

    ckpt = torch.load(weight, map_location=device)
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Image transforms
    img_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([config['dataset']['max_height'], config['dataset']['max_width']]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = MaskDetectionDataset(
        csv_file = config['dataset']['test_dir'],
        label_file = config['dataset']['label_file'],
        img_transform=img_transform)
    # print(config['dataset']['test_dir'])
    # exit()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)
    # ckpt_dir = os.path.join(config['train']['root_ckpt_dir'], config['train']['ckpt_dir'])
    # writer = SummaryWriter(ckpt_dir)
    # log = open(os.path.join(ckpt_dir, 'test_log.txt'), 'wt')

    # Network
    model = CNNModel(
        number_class=config['dataset']['num_classes']).to(device)
    model.load_state_dict(ckpt['model'])

    # Evaluate
    model.eval()
    total_loss = AverageMeter('Loss', ':.4e')
    total_acc = AverageMeter('Accuracy', ':6.2f')
    progress_bar = tqdm(test_loader, desc=f'Test')
    total_cm = np.array([[0,0],[0,0]])
    for batch_idx, batch in enumerate(progress_bar):
        images, labels = batch[0].to(device), batch[1].to(device)
        batch_size = images.size(0)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss.update(loss.item(), batch_size)
        _, predicted = torch.max(outputs, 1)
        # print('label: ', labels.cpu().numpy())
        # print('cm: ', )
        batch_cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
        total_cm += batch_cm
        acc = (labels==predicted).sum().item() / batch_size
        total_acc.update(acc, batch_size)
        progress_bar.set_postfix(loss=loss.item(), acc=acc)
    print('cm: ', total_cm)
    # log.write(f'\tTest loss: {total_loss.avg:.4f}\n\tTest acc: {total_acc.avg:.4f}\n')
    print(f'\tTest loss: {total_loss.avg:.4f} \tTest acc: {total_acc.avg:.4f}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--weight')
    args = parser.parse_args()
    start_time = time.time()
    evaluate(args.weight, args.config)
    print(time.time()-start_time)
