import os
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter


class Trainer(object):
    def __init__(self, device, model_name,  model, train_loader, val_loader, criterion, optimizer, \
        lr_scheduler, config, root_ckpt_dir='checkpoint', ckpt_dir=None, log_steps=10):
        self.device = device
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.log_steps = log_steps

        self.start_epoch = 0
        self.n_epochs = config['train']['n_epochs']
        self.clip = config['train']['clip']
        self.best_acc = 0.0

        if ckpt_dir is None:
            current_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            ckpt_dir = os.path.join(root_ckpt_dir, current_time)
        self.ckpt_dir = os.path.join(root_ckpt_dir, ckpt_dir)
        self.writer = SummaryWriter(self.ckpt_dir)
        self.log = open(os.path.join(self.ckpt_dir, 'log.txt'), 'wt')


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = AverageMeter('Loss', ':.4e')
        total_acc = AverageMeter('Accuracy', ':6.2f')
        progress_bar = tqdm(self.train_loader, desc=f'Train')
        for batch_idx, batch in enumerate(progress_bar):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            batch_size = images.size(0)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            #for param in self.model.parameters():
            #    print(param.grad.data.sum())
            self.optimizer.step()
            total_loss.update(loss.item(), batch_size)
            _, predicted = torch.max(outputs, 1)
            acc = (labels==predicted).sum().item() / batch_size
            total_acc.update(acc, batch_size)
            self.writer.add_scalar(f'Train/Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)
            self.writer.add_scalar(f'Train/Acc', acc, epoch * len(self.train_loader) + batch_idx)
            progress_bar.set_postfix(loss=loss.item(), acc=acc)
        self.log.write(f'\tTrain loss: {total_loss.avg:.4f}\n\tTrain acc: {total_acc.avg:.4f}\n')
        print(f'\tTrain loss: {total_loss.avg:.4f} \tTrain acc: {total_acc.avg:.4f}')

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.model.eval()
        total_loss = AverageMeter('Loss', ':.4e')
        total_acc = AverageMeter('Accuracy', ':6.2f')
        progress_bar = tqdm(self.val_loader, desc=f'Validation')
        for batch_idx, batch in enumerate(progress_bar):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            batch_size = images.size(0)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss.update(loss.item(), batch_size)
            _, predicted = torch.max(outputs, 1)
            acc = (labels==predicted).sum().item() / batch_size
            total_acc.update(acc, batch_size)
            self.writer.add_scalar(f'Validation/Loss', loss.item(), epoch * len(self.val_loader) + batch_idx)
            self.writer.add_scalar(f'Validation/Acc', acc, epoch * len(self.val_loader) + batch_idx)
            progress_bar.set_postfix(loss=loss.item(), acc=acc)
        self.log.write(f'\tValidation loss: {total_loss.avg:.4f}\n\tValidation acc: {total_acc.avg:.4f}\n')
        self.log.write(f"\tLearning rate: {self.optimizer.param_groups[0]['lr']}\n")
        print(f'\tValidation loss: {total_loss.avg:.4f} \tValidation acc: {total_acc.avg:.4f}')
        return total_loss.avg, total_acc.avg


    def save_checkpoint(self, epoch, val_acc):
        to_save = {
            'epoch': epoch,
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.log.write(f'Accuracy is improved from {self.best_acc:.4f} to {val_acc:.4f}. Saving weights...\n')
            print('save to : ', os.path.join(self.ckpt_dir, f'best_epoch_{epoch}_acc_{self.best_acc:.4f}.pt'))
            torch.save(to_save, os.path.join(self.ckpt_dir, f'best_epoch_{epoch}_acc_{self.best_acc:.4f}.pt'))
        else:
            torch.save(to_save, os.path.join(self.ckpt_dir, 'last.pt'))
            self.log.write(f'Accuracy is not improved from {self.best_acc:.4f}.\n')


    def resume_checkpoint(self, path):
        print('Resuming from checkpoint:', path)
        self.ckpt_dir = path.split('/')[0]
        ckpt = torch.load(path, map_location=self.device)
        self.start_epoch = ckpt['epoch']
        self.config = ckpt['config']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])


    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            self.log.write(f'{"-"*10} [Epoch {epoch+1}/{self.n_epochs}] {"-"*10}\n')
            print(f'{"-"*10} [Epoch {epoch+1}/{self.n_epochs}] {"-"*10}')
            self.train_epoch(epoch)
            val_avg_loss, val_avg_acc = self.eval_epoch(epoch)
            self.lr_scheduler.step()
            self.save_checkpoint(epoch, val_avg_acc)
        self.writer.close()
        self.log.close()
