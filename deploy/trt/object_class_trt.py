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
from deploy.trt.trt_loader import TrtCNN
from data.augment import get_img_transform

class ObjectClassName:
    def __init__(self, config):
        # Network
        print('[INFO] Load model TrtCNN')
        self.net = TrtCNN(config['weight'])
        self.net.build()
        print('Finished loading model!')
        
        self.img_transform = transforms.Compose([
            # transforms.ToPILImage(),
            get_img_transform(config['model']['dataset'], False)
        ])

    def predict(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.img_transform(img)
        img.unsqueeze_(dim=0)
        img = Variable(img)
        img = img.numpy()

        outputs = self.net.run(img)
        outputs = torch.Tensor(outputs)
        outputs = F.softmax(outputs, dim=-1)
        _, predicted = torch.max(outputs, 1)
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
        img = img.numpy()
        outputs = self.net.run(img)
        outputs = torch.Tensor(outputs)
        outputs = F.softmax(outputs, dim=-1)
        _, predicted = torch.max(outputs, 1)
        print('predicted: ', list(outputs[:, 0]))
        return [i.item() for i in predicted]