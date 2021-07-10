import yaml
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import random
import os
import shutil
from PIL import Image, ImageFilter

from dataset import MaskDetectionDataset
from model.cnn_mobilenet import CNNModel
from utils import set_seed, set_determinism, count_parameters
from trainer import Trainer


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

class RandomGaussBlur(object):
    """Random GaussBlurring on image by radius parameter.
    Args:
        radius (list, tuple): radius range for selecting from; you'd better set it < 2
    """
    def __init__(self, radius=None):
        if radius is not None:
            assert isinstance(radius, (tuple, list)) and len(radius) == 2, \
                "radius should be a list or tuple and it must be of length 2."
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius={0})'.format(self.radius)


def train(args):
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Image transforms
    img_transform = transforms.Compose([
            RandomGaussBlur(radius=[-3.0, 3.0]),
            transforms.Resize([config['dataset']['max_height'], config['dataset']['max_width']]),
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0, hue=0),
            transforms.ToTensor(),
            # AddGaussianNoise(0.0, 0.05),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = MaskDetectionDataset(
        csv_file = config['dataset']['train_dir'],
        label_file = config['dataset']['label_file'],
        img_transform=img_transform)

    val_dataset = MaskDetectionDataset(
        csv_file = config['dataset']['val_dir'],
        label_file = config['dataset']['label_file'],
        img_transform=img_transform)

    set_seed(config['train']['seed'])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True)

    set_seed(config['train']['seed'])
    val_loader = DataLoader(dataset=val_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True)

    class_weights = train_dataset.get_classweight()
    print('class weight: ', class_weights)

    ckpt_dir = os.path.join(config['train']['root_ckpt_dir'], config['train']['ckpt_dir'])
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    infor_log = open(os.path.join(ckpt_dir, 'infor.txt'), 'wt')
    print('infor: ', train_dataset.get_infor())
    infor_log.write(train_dataset.get_infor())

    shutil.copyfile(config_path, os.path.join(ckpt_dir, 'config.yaml'))

    # Network
    model = CNNModel(
        number_class=config['dataset']['num_classes']).to(device)
    count_parameters(model)

    # Loss
    set_seed(config['train']['seed'])
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Optimizer & Scheduler
    set_seed(config['train']['seed'])
    optimizer = optim.Adam(
        lr=config['optimizer']['start_lr'],
        weight_decay=config['optimizer']['weight_decay'],
        params=filter(lambda p: p.requires_grad, model.parameters())
        )
    # optimizer = optim.SGD(
    #      params=filter(lambda p: p.requires_grad, model.parameters()),
    #      lr=config['optimizer']['start_lr'],
    #      momentum=0.9)
    set_seed(config['train']['seed'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['optimizer']['step_reduce_lr'], gamma=0.1)

    # Train
    set_seed(config['train']['seed'])
    trainer = Trainer(
        device=device,
        model_name=config['model']['model_name'],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        root_ckpt_dir=config['train']['root_ckpt_dir'],
        ckpt_dir=config['train']['ckpt_dir'])

    if args.resume:
        trainer.resume_checkpoint(args.resume)
        
    set_seed(config['train']['seed'])
    set_determinism()
    trainer.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, help='path of pretrained')
    args = parser.parse_args()
    train(args)
