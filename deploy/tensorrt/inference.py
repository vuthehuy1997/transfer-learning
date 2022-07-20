from PIL import Image
import time
import numpy as np
import cv2
from trt_loader import TrtCNN
from albumentations import (Compose,Resize,)
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor
from torchvision import models
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
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data


def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    # print('confidences: ', confidences)
    # indices = np.asarray(indices)
    # print('indices: ', indices)

    while confidences[indices[0][i]] > 20:
        print(confidences[indices[0][i]])
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1

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
    
    pytorch_model = models.resnet50(pretrained=True)
    # inference stage --------------------------------------------------------------------------------------------------
    pytorch_model.eval()
    pytorch_model.cuda()

    # trt_model = Trt('resnet50.trt')
    
    filenames = ["dog.jpg", "turkish_coffee.jpg"]
    i = 0
    total = 0
    while i < 10:
        i+=1
    # pytorch
        input = preprocess_image(filenames[i%2]).cuda()
        s = time.time()
        output = pytorch_model(input)
        time_ = time.time() - s
        total += time_
        print('pytorch: ',time_)
        postprocess(output)
    # Trt
        # img = preprocess_image(filenames[i%2]).numpy()
        # s = time.time()
        # output = trt_model.predict(img)
        # time_ = time.time() - s
        # total += time_
        # print('tensorrt: ',time_)
        # output = torch.Tensor(output)
        # postprocess(output)

    print('total: ',total)
