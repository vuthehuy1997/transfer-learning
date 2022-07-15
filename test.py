import os
import time
import yaml
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
import datetime
from torch.utils.data import DataLoader
import math
from utils import AverageMeter

from data.dataset import TFDataset
from model.network.cnn import CNNModel
from data.augment import get_img_transform
from config.config import get_config

def evaluate(args):
    config = get_config()
    print(config)

    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))


    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt['config']

    config_data = yaml.load(open(args.data, 'r'), Loader=yaml.Loader)
    config.update(config_data)

    print(config)

    

    # Image transforms
    img_transform = get_img_transform(config['model']['dataset'], False)

    test_dataset = TFDataset(
        label_dir = config['data']['test_dir'],
        label_file = config['data']['test_label'],
        label_name = config['data']['label_name'],
        img_transform=img_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)
    ckpt_dir = os.path.join(config['train']['root_ckpt_dir'], config['train']['ckpt_dir'])
    writer = SummaryWriter(ckpt_dir)
    log = open(os.path.join(ckpt_dir, 'test_log.txt'), 'wt')

    # Network
    model = CNNModel(
        fe_name=config['model']['cnn']['module'], version=config['model']['cnn']['version'],
        feature_extract=config['model']['cnn']['feature_extract'], pretrained=config['model']['cnn']['pretrained'],
        number_class=config['data']['num_classes'], drop_p=config['regularization']['dropout']).to(device)
    model.load_state_dict(ckpt['model'])

    # Evaluate
    model.eval()
    total_loss = AverageMeter('Loss', ':.4e')
    total_acc = AverageMeter('Accuracy', ':6.2f')
    progress_bar = tqdm(test_loader, desc=f'Test')
    for batch_idx, batch in enumerate(progress_bar):
        images, labels = batch[0].to(device), batch[1].to(device)
        batch_size = images.size(0)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss.update(loss.item(), batch_size)
        _, predicted = torch.max(outputs, 1)
        acc = (labels==predicted).sum().item() / batch_size
        total_acc.update(acc, batch_size)
        progress_bar.set_postfix(loss=loss.item(), acc=acc)
    log.write(f'\tTest loss: {total_loss.avg:.4f}\n\tTest acc: {total_acc.avg:.4f}\n')
    print(f'\tTest loss: {total_loss.avg:.4f} \tTest acc: {total_acc.avg:.4f}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--ckpt')
    args = parser.parse_args()
    start_time = time.time()
    evaluate(args)
    print(time.time()-start_time)
