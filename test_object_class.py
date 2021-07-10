import os
from os import listdir, mkdir
from os.path import isfile, join, basename, exists
from shutil import copyfile
import time
import argparse
import json

import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch

from object_class import ObjectClassName
from config_api import gpu_object_class
if gpu_object_class == -1:
    DEVICE = 'cpu'
else:
    DEVICE = 'cuda:' + str(gpu_object_class)
device = torch.device(DEVICE)

out_root = './test_result'
if not exists(out_root):
    mkdir(out_root)
label_file = 'labels.json'

def evaluate(weight, csv_file):
    maskdetection = MaskDetection(weight)
    threshold = 0.5

    #---------------------get output folder
    out_folder = join(out_root, os.path.splitext(csv_file)[0].\
    replace('./', '').replace('/','_'))
    if not exists(out_folder):
        mkdir(out_folder)
    out_folder = join(out_folder, os.path.splitext(weight)[0].\
    replace('./', '').replace('/','_'))
    if not exists(out_folder):
        mkdir(out_folder)
    print('out: ', out_folder)

    #-----------------------read label json
    data_labels = []
    f = open(label_file,)
    json_labels = json.load(f)
    print('json_labels: ', json_labels)
    for json_label in json_labels:
        data_labels.append(json_labels[json_label])
    print('data_labels: ', data_labels)

    #-----------------------read data test
    df = pd.read_csv(csv_file)
    paths = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    int_labels = [data_labels.index(x) for x in labels]
    print('int_labels: ', int_labels)
    # exit()
    int_predicts = []

    for path in paths:
        # print('path: ', path)
        face = cv2.imread(path)
        predict =  maskdetection.predict(face)
        # print('predict: ', predict)
        
        if predict > threshold:
            int_predicts.append(0)
        else:
            int_predicts.append(1)
    #---------------save wrong image
    for path, label, predict in zip(list(paths), list(int_labels), int_predicts):
        print('path:', path)
        print('label:', label)
        print('predict:', predict)
        out_wrong = join(out_folder, str(label))
        if not exists(out_wrong):
            mkdir(out_wrong)
        if predict != label:
            copyfile(path, join(out_wrong, basename(path)))
        # exit()

    #---------------eval
    cm = confusion_matrix(int_labels, int_predicts)
    tn, fp, fn, tp = cm.ravel()
    print('cm: ', cm)
    acc = (tp+tn)/(tp+tn+fp+fn)
    print('acc: ', acc)


    #---------------to file
    predicts = [data_labels[int(x)] for x in int_predicts]
    df['predict'] = predicts
    df.to_csv(join(out_folder, 'result.csv'))

    log = open(join(out_folder, 'result.txt'), 'wt')
    log.write(f'acc: {acc:.4f}\n')
    log.write(f'confusion matrix:\n {tn}\t {fp}\n {fn}\t {tp}\n')
    # exit()
    
    # os.path.splitext(file_name)[0] + '_augmentation.jpg')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight')
    parser.add_argument('--csv_file', type=str, \
    default='./dataset/dataset_test/Dataset/test.csv', \
    help='path to image file')
    args = parser.parse_args()
    start_time = time.time()
    
    evaluate(args.weight, args.csv_file)
    print(time.time()-start_time)