import yaml
import argparse
from collections import Counter
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


from utils import set_seed, set_determinism, count_parameters
from trainer import Trainer
from model.network.cnn import CNNModel

from config.config import get_config
from model.loss.loss import get_loss_from_config
from model.optimizer.optimizer import get_optimizer_from_config
from data.augment import get_img_transform



def train(args):
    config = get_config(args.config)
    print(config)

    if config['classification']:
        from data.dataset import TFDataset
    else:
        from data.dataset_regression import TFDataset

    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))


    # Image transforms
    img_transform = get_img_transform(config['model']['dataset'], config['aug'])

    train_dataset = TFDataset(
        label_dir = config['data']['train_dir'],
        label_file = config['data']['train_label'],
        label_name = config['data']['label_name'],
        img_transform=img_transform)

    val_dataset = TFDataset(
        label_dir = config['data']['val_dir'],
        label_file = config['data']['val_label'],
        label_name = config['data']['label_name'],
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

    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Network
    model = CNNModel(
        fe_name=config['model']['cnn']['module'], version=config['model']['cnn']['version'],
        feature_extract=config['model']['cnn']['feature_extract'], pretrained=config['model']['cnn']['pretrained'],
        number_class=config['data']['num_classes'], drop_p=config['regularization']['dropout']).to(device)
    count_parameters(model)

    # Loss
    set_seed(config['train']['seed'])
    criterion = get_loss_from_config(config['loss'],class_weight=class_weights).to(device)

    # Optimizer & Scheduler
    set_seed(config['train']['seed'])
    optimizer = get_optimizer_from_config(config['optimizer'],filter(lambda p: p.requires_grad, model.parameters()))
    
    set_seed(config['train']['seed'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step_reduce_lr'], gamma=0.1)

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
