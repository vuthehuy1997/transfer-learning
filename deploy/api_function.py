import os
from os import listdir, mkdir
from os.path import isfile, join, basename, exists, isdir
import shutil 
import time
from shutil import copyfile
import threading
import json
from random import choice
from io import BytesIO
import sys

import insightface
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import requests
from flask import send_file, jsonify
import jwt
import torchvision.transforms as transforms
import mxnet as mx

from config.config import data_path, process_data_path, test_embedding_threshold, tree_path, \
                            search_tree, dataset_embedding_threshold, gpu_insight_face, gpu_retinaface
from config.constant import Folders, Test_Folders, Phases_choose_image, Tasks_choose_image
from utils import *
import insightface
from insightface.utils import face_align
# from insightface_mxnet.insightface import InsightFace


# retinaface = insightface.model_zoo.get_model('retinaface_r50_v1')
# retinaface.prepare(ctx_id=gpu_retinaface, nms=0.4)
desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def jsonify_str(app, output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result

def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}

def create_query_result(result, error=None):
    if error is not None:
        results = {'result':None, 'error':str(error)}
    else:
        results = {'result':result, 'error':''}
    query_result = {
        'version': 1,
        'results': results
    }
    return query_result

def detect_mask_service(img, ratio, maskdetection, retinaface):
    a = time.time()
    # bboxes, faces = dsfd.detect_multi(img, limit=None)
    # bboxes, faces, landmark5s = retinaface.detect_align(img, remove_no_landmark = True)
    bboxes, pts5s = retinaface.detect(img, threshold=0.5)
    print('bboxes: ', bboxes)
    ########################### process get face image
    faces = []
    # for pts5 in pts5s:
    #     nimg = face_align.norm_crop(img, pts5)
    #     faces.append(nimg)

    bboxes = [list(bbox) for bbox in bboxes]
    if len(bboxes) == 0:
        return [], [], []
    for i in range(len(bboxes)):
        bboxes[i][1] = int(max(0, bboxes[i][1]))
        bboxes[i][0] = int(max(0, bboxes[i][0]))
        bboxes[i][3] = int(min(img.shape[0], bboxes[i][3]))
        bboxes[i][2] = int(min(img.shape[1], bboxes[i][2]))
    for bbox in bboxes:
        faces.append(img[bbox[1]: bbox[3],bbox[0]: bbox[2]])
        # faces.append(img[max(0, bbox[1]-5): min(img.shape[0], bbox[3]+5),max(0, bbox[0] - 5): min(img.shape[1], bbox[2] +5)])

    for i in range(len(faces)):
        cv2.imwrite('./tmps/test_' + str(i) + '.jpg', cv2.resize(faces[i], (32,32), interpolation=cv2.INTER_LINEAR))

    ############################
    # bboxes, faces = ssh.detect_align(img)
    # bboxes, faces = mtcnn.align_multi(img)
    b = time.time()
    predicteds = []
    predicteds = maskdetection.predict_batch(faces)


    # for i in range(len(bboxes)):
    #     rs =  maskdetection.predict(faces[i])
    #     predicteds.append(rs)
        # cv2.imwrite('test_face.jpg', faces[i])
        
    c = time.time()
    time_detect = b-a
    time_mask = c-b

    print('len of faces: ', len(faces))
    print('time detect: ', time_detect)
    print('time mask: ', time_mask)
    print('mask: ', predicteds)

    img = cv2.rectangle(img,(int(bboxes[0][0]),int(bboxes[0][1])),(int(bboxes[0][2]), int(bboxes[0][3])),(0,0,255),2)
    # cv2.imwrite('./tmps/tmp.jpg', img)
    if len(bboxes) == 0:
        print('no face')
        return [], [], []
    else:
        bboxes = np.array(bboxes)/ratio
        pts5s = np.array(pts5s)/ratio
        # print('bboxes 2: ', bboxes)
        print('ratio: ', ratio)
    
    
    return [bbox[:4].tolist() for bbox in bboxes], \
    [pts5.tolist() for pts5 in pts5s], predicteds

