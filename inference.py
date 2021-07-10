import os
import time
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PIL import Image
import datetime
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

def evaluate(weight, image_path):

    ckpt = torch.load(weight, map_location=device)
    config = ckpt['config']

    if config['dataset']['max_height'] == 224:
        from model.cnn import CNNModel
    elif config['dataset']['max_height'] == 32:
        from model.cnn_simple import CNNModel
    # Image transforms
    img_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize([config['dataset']['max_height'], config['dataset']['max_width']]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Network
    model = CNNModel(
        number_class=config['dataset']['num_classes']).to(device)
    model.load_state_dict(ckpt['model'])

    # Evaluate
    model.eval()
    img = Image.open(image_path)
    img = img_transform(img)
    img.unsqueeze_(dim=0)
    img = Variable(img)
    img = img.to(device)
    
    outputs = model(img)
    outputs = F.softmax(outputs, dim=-1)
    print('outputs: ', outputs)
    _, predicted = torch.max(outputs, 1)
    print('predicted: ', predicted.item())
    print('predicted: ', outputs[0][predicted].item())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight')
    parser.add_argument('--image_path', type=str, \
    default='/storage/face_recognition/face-recognize-checkin/Mask_Detection/Dataset/test/1/maksssksksss1_0.jpg', help='path to image file')
    args = parser.parse_args()
    start_time = time.time()
    evaluate(args.weight, args.image_path)
    print(time.time()-start_time)
