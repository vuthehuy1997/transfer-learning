import os
from PIL import Image
import shutil 
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import cv2
import torch
import time
import os
import sys
import numpy as np

# from utils import *
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from flask import Flask, jsonify, request
import requests
import time
from flask_cors import CORS, cross_origin
from random import choice
from io import BytesIO
from flask import send_file
import concurrent.futures
from os import listdir, mkdir
from os.path import isfile, join, basename, exists, isdir
import threading
import json
import jwt
import time
import zlib
import sys

from config_api import gpu_mask_detection, gpu_retinaface
from api_function import random_headers, create_query_result, \
detect_mask_service, jsonify_str

from mask_detection import MaskDetection
# from inference import FaceMaskClassifier
import insightface

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSON_AS_ASCII'] = False

retinaface = insightface.model_zoo.get_model('retinaface_r50_v1')
retinaface.prepare(ctx_id=gpu_retinaface)
# retinaface = RetinaFacePytorch(confidence_threshold = 0.2, nms_threshold = 0.4)

maskdetection = MaskDetection()
# maskdetection = FaceMaskClassifier('FaceMaskDetection/results/cp.ckpt')
# detector = RetinaFaceCoV('insightface_master/detection/RetinaFaceAntiCov/model/mnet_cov2', 0, gpu_retinaface, 'net3l')
no_mask_thresh = 0.5

def download_image(image_url):
    header = random_headers()

    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)

    image = Image.open(BytesIO(response.content)).convert('RGB')

    return image
def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def draw_mask(bbox, frame, mask):
    bbox = [int(i) for i in bbox]
    frame = np.array(frame)
    start = (int(bbox[0]),int(bbox[1]))
    end = (int(bbox[2]),int(bbox[3]))

    if mask > no_mask_thresh:
        color = (0, 0, 255)
        label = 'NoMask'
    else:
        color = (0, 255, 0)
        label = 'Mask'
        mask = 1-mask

    color_a = np.append(color,[0])
    color_a = tuple([int(i) for i in color_a])
    thickless = 1
    frame = cv2.rectangle(frame,start,end,color,thickless)

    fontpath = "./AndikaNewBasic-R.ttf" # font path
    font = ImageFont.truetype(fontpath, 15)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    draw.text((bbox[0],bbox[1]-30),  label + ': ' + "{:.3f}".format(mask), font = font, fill = color_a)

    frame = np.array(img_pil)
    return frame

def draw_box(bbox, pts5, frame):
    bbox = [int(i) for i in bbox]
    frame = np.array(frame)
    start = (int(bbox[0]),int(bbox[1]))
    end = (int(bbox[2]),int(bbox[3]))
    color = (0, 255, 0)
    color_a = np.append(color,[0])
    color_a = tuple([int(i) for i in color_a])
    thickless = 1
    frame = cv2.rectangle(frame,start,end,color,thickless)

    for pts in pts5:
        print('pts: ', pts)
        frame = cv2.circle(frame, tuple(map(int, pts)), 2, (0, 0, 255), 2)
    return frame

#############################################################################################

@app.route("/face_mask", methods=['GET','POST'])
@cross_origin()
def face_mask():
    print('call api face_recognize')
    print('worker: ', str(os.getpid()))
    start_all = time.time()
    if request.method == "POST":
        try:
            print('request:', request)
            print('receive image binary string')
            # dataa = base64.b64decode(request.data)
            nparr = np.fromstring(request.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as ex:
            print(ex)
            return jsonify(create_query_result("", "Upload error or Can not find image path or Can not open image"))
    else:
        try:
            image_url = request.args.get('url', default='', type=str)
            print('url: ',image_url)
            img = download_image(image_url)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
        except Exception as ex:
            print("Can not download image: ", ex)
            return jsonify(create_query_result("", "Wrong parameter or Can not download image"))
    # start = time.time()
    pre_shape = img.shape
    ratio = 1.0
    if pre_shape[0] > 512:
        ratio = 512.0 / pre_shape[0]
        img_small = cv2.resize(img, (int(img.shape[1]*ratio),int(img.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    # print("type img: ", type(img))
    # print("shape img: ", img.shape)

    start = time.time()
    bboxes, pts5s, masks = detect_mask_service(img_small, ratio, maskdetection, retinaface)
    print('bboxes after process: ', bboxes)
    # img = cv2.resize(img, (int(img.shape[1]/ratio),int(img.shape[0]/ratio)), interpolation=cv2.INTER_AREA)
    img_tmp = img.copy()
    for idx in range(len(bboxes)):
        # print('idx: ', idx)
        img_tmp = draw_box(bboxes[idx], pts5s[idx], img_tmp)
    cv2.imwrite('./tmps/test_box.jpg', img_tmp)

    for idx in range(len(bboxes)):
        # print('idx: ', idx)
        img = draw_mask(bboxes[idx], img, masks[idx])
    pil_im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    end = time.time()
    print('processing time: ', end-start)

    with torch.cuda.device('cuda:'+str(gpu_mask_detection)):
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda:'+str(gpu_retinaface)):
        torch.cuda.empty_cache()

    return serve_pil_image(pil_im)



@app.route("/face_mask/get_bbox", methods=['GET','POST'])
@cross_origin()
def get_bbox():
    print('call api face_recognize')
    print('worker: ', str(os.getpid()))
    start_all = time.time()
    if request.method == "POST":
        try:
            print('request:', request)
            print('receive image binary string')
            # dataa = base64.b64decode(request.data)
            nparr = np.fromstring(request.data, np.uint8)
            # decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as ex:
            print(ex)
            return jsonify(create_query_result("", "Upload error or Can not find image path or Can not open image"))
    else:
        try:
            image_url = request.args.get('url', default='', type=str)
            print('url: ',image_url)
            img = download_image(image_url)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
        except Exception as ex:
            print("Can not download image: ", ex)
            return jsonify(create_query_result("", "Wrong parameter or Can not download image"))
    # start = time.time()
    pre_shape = img.shape
    ratio = 1.0
    if pre_shape[0] > 1024:
        ratio = 1024.0 / pre_shape[0]
        img_small = cv2.resize(img, (int(img.shape[1]*ratio),int(img.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()

    start = time.time()
    cv2.imwrite('./tmps/test_img.jpg', img_small)
    bboxes, pts5s, masks = detect_mask_service(img_small, ratio, maskdetection, retinaface)
    print('bboxes after process: ', bboxes)
    results = []
    for bbox, pts5, mask in zip(bboxes, pts5s, masks):
        results.append({'bbox': bbox, 'pts5s': pts5, 'is_mask': str(mask)})


    end = time.time()
    print('processing time: ', end-start)

    with torch.cuda.device('cuda:'+str(gpu_mask_detection)):
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda:'+str(gpu_retinaface)):
        torch.cuda.empty_cache()

    return jsonify_str(app, create_query_result(results, None))

@app.route("/face_mask/get_bbox_local", methods=['GET'])
@cross_origin()
def get_bbox_local():
    print('call api face_recognize')
    print('worker: ', str(os.getpid()))
    start_all = time.time()
    if request.method == "GET":
        try:
            image_path = request.args.get('path', default='', type=str)
            print('path: ',image_path)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
        except Exception as ex:
            print("Can not download image: ", ex)
            return jsonify(create_query_result("", "Wrong parameter or Can not download image"))
    # start = time.time()
    pre_shape = img.shape
    ratio = 1.0
    if pre_shape[0] > 1024:
        ratio = 1024.0 / pre_shape[0]
        img_small = cv2.resize(img, (int(img.shape[1]*ratio),int(img.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    print('image_small size: ', img_small.shape)
    start = time.time()
    cv2.imwrite('./tmps/test_img.jpg', img_small)
    bboxes, pts5s, masks = detect_mask_service(img_small, ratio, maskdetection, retinaface)
    print('bboxes after process: ', bboxes)
    results = []
    for bbox, pts5, mask in zip(bboxes, pts5s, masks):
        results.append({'bbox': bbox, 'pts5s': pts5, 'is_mask': str(mask)})
        
    end = time.time()
    print('processing time: ', end-start)

    with torch.cuda.device('cuda:'+str(gpu_mask_detection)):
        torch.cuda.empty_cache()
    with torch.cuda.device('cuda:'+str(gpu_retinaface)):
        torch.cuda.empty_cache()

    return jsonify_str(app, create_query_result(results, None))
        
        
if __name__=='__main__':
    app.run("172.18.5.30", 1417, threaded=True, debug=False)
