import os
from os import listdir, mkdir
from os.path import isfile, join, basename, exists
from shutil import copyfile
import time
import argparse
import json
import yaml
from PIL import Image
# import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import torch

def evaluate(config_file, file_dir, csv_file, out_dir):
    config = yaml.load(open(config_file, 'r'), Loader=yaml.Loader)

    print(config['weight'])
    if config['weight'].endswith('.pt'): # Pytorch model
        from object_class import ObjectClassName
        model = ObjectClassName(config)
    elif config['weight'].endswith('.trt'): # Tensorrt model
        from deploy.trt.object_class_trt import ObjectClassName
        model = ObjectClassName(config)

    
    threshold = 0.5

    #---------------------get output folder
    if not exists(out_dir):
        mkdir(out_dir)
    # print('out: ', out_dir)

    #-----------------------read label json
    data_labels = []
    f = open(config['data']['label_name'],)
    json_labels = json.load(f)
    # print('json_labels: ', json_labels)
    for json_label in json_labels:
        data_labels.append(json_labels[json_label])
    # print('data_labels: ', data_labels)

    #-----------------------read data test
    df = pd.read_csv(csv_file)
    paths = df.iloc[:, 0]
    paths = [os.path.join(file_dir, path) for path in paths]
    labels = df.iloc[:, 1]
    int_labels = [data_labels.index(str(x)) for x in labels]
    # print('int_labels: ', int_labels)
    # exit()
    int_predicts = []

    for path in paths:
        # print('path: ', path)
        # face = cv2.imread(path)
        image = Image.open(path)
        predict =  model.predict(image)
        # print('predict: ', predict)
        int_predicts.append(predict)
    #---------------save wrong image
    for path, label, predict in zip(list(paths), list(int_labels), int_predicts):
        # print('path:', path)
        # print('label:', label)
        # print('predict:', predict)
        
        if predict != label:
            out_wrong = join(out_dir, str(label), str(predict))
            if not exists(out_wrong):
                os.makedirs(out_wrong, exist_ok=True)
            copyfile(path, join(out_wrong, basename(path)))
        # exit()

    #---------------eval
    print('Number image: ', len(int_labels))
    cm = confusion_matrix(int_labels, int_predicts)
    print('cm: ', cm)
    
    acc = accuracy_score(int_labels, int_predicts)
    print('acc: ', acc)

    precision = precision_score(int_labels, int_predicts, average='micro')
    print('precision: ', precision)

    recall = recall_score(int_labels, int_predicts, average='micro')
    print('recall: ', recall)


    #---------------to file
    predicts = [data_labels[int(x)] for x in int_predicts]
    df['predict'] = predicts
    df.to_csv(join(out_dir, 'result.csv'))

    log = open(join(out_dir, 'result.txt'), 'wt')
    log.write(f'acc: {acc:.4f}\n')
    log.write(f'precision:\n {precision}\n')
    log.write(f'recall:\n {recall}\n')
    log.write(f'confusion matrix:\n {cm}\n')
    # exit()
    
    # os.path.splitext(file_name)[0] + '_augmentation.jpg')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--csv-file', type=str, \
        default='./dataset/dataset_8/public_test_review/public_test.csv', \
        help='path to image file')
    parser.add_argument('--data-dir', type=str, \
        default='./dataset/dataset_8/public_test_review', \
        help='path to image file')
    parser.add_argument('--out-dir', type=str, \
        default='./test_result/tmp', \
        help='path to image file')
    args = parser.parse_args()
    start_time = time.time()
    
    evaluate(args.config, args.data_dir, args.csv_file, args.out_dir)
    print(time.time()-start_time)