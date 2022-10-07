from PIL import Image
import time
import numpy as np
import cv2
import json
from model.network.cnn import CNNModel
from deploy.trt.trt_loader import TrtCNN
from albumentations import (Compose,Resize,)
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor
import torch
import time

def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    # prepare batch
    # batch_data = torch.unsqueeze(input_data, 0)
    batch_data = torch.stack([input_data,input_data])
    return batch_data


def postprocess(output_data):
    x =  output_data.cpu().detach().numpy()
    # print(x)
    print(x.shape)
    # get class names
    classes = []
    f = open('data/labels.json',)
    labels = json.load(f)
    # print('label: ', labels)
    for label in labels:
        classes.append(labels[label])
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    # print('confidences: ', confidences)
    # indices = np.asarray(indices)
    # print('indices: ', indices)

    # while confidences[indices[0][i]] > 50:
    #     print(confidences[indices[0][i]])
    #     class_idx = indices[0][i]
    #     print(
    #         "class:",
    #         classes[class_idx],
    #         ", confidence:",
    #         confidences[class_idx].item(),
    #         "%, index:",
    #         class_idx.item(),
    #     )
    #     i += 1

class Trt(object):
    def __init__(self, model):
        print('[INFO] Load model')
        self.model = TrtCNN(model)
        self.model.build()

    def predict(self, img, return_prob = True):
        '''
            Predict single-line image
            Input:
                - img: pillow Image
        '''
        tik = time.time()
        

        output = self.model.run(img)
        return output


if __name__ == '__main__':

    # load pre-trained model -------------------------------------------------------------------------------------------
    ckpt = torch.load('weights/resnet50.pt')
    config = ckpt['config']
    config['device'] = 'cuda:0'

    # Network
    pytorch_model = CNNModel(
        fe_name=config['model']['cnn']['module'], version=config['model']['cnn']['version'],
        feature_extract=config['model']['cnn']['feature_extract'], pretrained=config['model']['cnn']['pretrained'],
        number_class=config['data']['num_classes'], drop_p=config['regularization']['dropout']).to(config['device'])

    pytorch_model.load_state_dict(ckpt['model'])
    print('Finished loading model!')
    # print(pytorch_model)
    
    pytorch_model = pytorch_model.to(config['device'])
    pytorch_model.eval()

    trt_model = Trt('weights/resnet50.trt')
    
    filenames = ["dataset/dataset_8/public_test_review/images/brc_test_001_0_225.jpg", "dataset/dataset_8/public_test_review/images/brc_test_002_0_270.jpg"]
    i = 0
    total_pytorch = 0
    total_trt = 0
    while i < 10:
        i+=1
    # pytorch
        input = preprocess_image(filenames[i%2]).cuda()
        s = time.time()
        output = pytorch_model(input)
        time_ = time.time() - s
        total_pytorch += time_
        print('pytorch: ',time_)
        postprocess(output)
    # Trt
        img = preprocess_image(filenames[i%2]).numpy()
        print(img.shape)
        s = time.time()
        output = trt_model.predict(img)
        time_ = time.time() - s
        total_trt += time_
        print('tensorrt: ',time_)
        output = torch.Tensor(output)
        postprocess(output)

    print('total_pytorch: ',total_pytorch)
    print('total_trt: ',total_trt)
