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
import cv2
import datetime
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# from model.cnn_simple import CNNModel
# from model.cnn import CNNModel

from config_api import gpu_mask_detection
if gpu_mask_detection == -1:
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda:' + str(gpu_mask_detection)

class ObjectClassName:
    def __init__(self, weight = './checkpoint/checkpoint_all_6_no_feature_extract_simple_randomgauss/last.pt'):
        # torch.set_grad_enabled(False)
        # net and model
        self.device = torch.device(DEVICE)

        ckpt = torch.load(weight, map_location=self.device)
        config = ckpt['config']

        # Network
        if config['model']['name'] == 'resnet18':
            from model.cnn_resnet18 import CNNModel
        elif config['model']['name'] == 'mobilenet_v2':
            from model.cnn_mobilenet_v2 import CNNModel
        elif config['model']['name'] == 'cnn_simple_32':
            from model.cnn_simple_32 import CNNModel
        elif config['model']['name'] == 'cnn_simple_64':
            from model.cnn_simple_64 import CNNModel

        self.net = CNNModel(
            number_class=config['dataset']['num_classes']).to(self.device)
        self.net.load_state_dict(ckpt['model'])
        print('Finished loading model!')
        # print(net)
        cudnn.benchmark = False
        
        self.net = self.net.to(self.device)
        self.net.eval()
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([config['dataset']['max_height'], config['dataset']['max_height']]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        return outputs[0][0].item()

    def predict_batch(self, imgs):
        tensor_imgs = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        return [output[0].item() for output in outputs]