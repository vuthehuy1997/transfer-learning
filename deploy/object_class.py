import os
import time
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
# import cv2
import datetime
from model.network.cnn import CNNModel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from data.augment import get_img_transform

class ObjectClassName:
    def __init__(self, weight = './last.pt', DEVICE = 'cuda:0'):
        # torch.set_grad_enabled(False)
        # net and model
        self.device = torch.device(DEVICE)

        ckpt = torch.load(weight, map_location=self.device)
        config = ckpt['config']

        # Network
        self.net = CNNModel(
            fe_name=config['model']['cnn']['module'], version=config['model']['cnn']['version'],
            feature_extract=config['model']['cnn']['feature_extract'], pretrained=config['model']['cnn']['pretrained'],
            number_class=config['data']['num_classes'], drop_p=config['regularization']['dropout']).to(self.device)

        self.net.load_state_dict(ckpt['model'])
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = False
        
        self.net = self.net.to(self.device)
        self.net.eval()
        self.img_transform = transforms.Compose([
            # transforms.ToPILImage(),
            get_img_transform(config['model']['dataset'], False)
        ])

    def predict(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(img)
        img.unsqueeze_(dim=0)
        img = Variable(img)
        img = img.to(self.device)
        with torch.no_grad():
            outputs = self.net(img)
            outputs = F.softmax(outputs, dim=-1)
            # print('outputs: ', outputs)
            _, predicted = torch.max(outputs, 1)
            # print('predicted: ', predicted.item())
            # print('predicted: ', outputs[0][predicted].item())
        # return predicted.item(), outputs[0][predicted].item()
        # return outputs[0][0].item()
        return predicted.item()

    def predict_batch(self, imgs):
        tensor_imgs = []
        for img in imgs:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.img_transform(img)
            tensor_imgs.append(img)
        tensor_imgs = torch.stack(tensor_imgs)
        # img.unsqueeze_(dim=0)
        img = Variable(tensor_imgs)
        img = img.to(self.device)
        with torch.no_grad():
            outputs = self.net(img)
            outputs = F.softmax(outputs, dim=-1)
            # print('outputs: ', outputs)
            _, predicted = torch.max(outputs, 1)
            print('predicted: ', list(outputs[:, 0]))
            # print('predicted: ', predicted.item())
            # print('predicted: ', outputs[0][predicted].item())
        # return [output[0].item() for output in outputs]
        return [i.item() for i in predicted]