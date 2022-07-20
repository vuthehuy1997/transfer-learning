import random
from collections import Counter

import torch

from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import json
import pandas as pd
from PIL import Image


class TFDataset(Dataset):
    """
    TF image  ...
    """
    def __init__(self, label_dir, label_file , label_name=None, img_transform=None):
        """
        Args:
            label_dir: data path
            label_file: file contains data path and label
            label_name: file mapping label
            img_transform: img transform
        """
        data_labels = []
        f = open(label_name,)
        labels = json.load(f)
        print('label: ', labels)
        for label in labels:
            data_labels.append(labels[label])
        self.data_labels = data_labels

        df = pd.read_csv(os.path.join(label_dir, label_file))
        self.dir = label_dir
        self.paths = df.iloc[:, 0]
        self.labels = df.iloc[:, 1]
        self.labels = list(map(str, self.labels))
        self.img_transform = img_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.paths[idx])
        # print('img_path: ', img_path)
        image = Image.open(img_path)
        # print('image: ', image)
        if self.img_transform != None:
            image = self.img_transform(image)
        
        label = torch.tensor(float(self.labels[idx]))
        return image, label
    
    def get_classweight(self):
        return

    def get_infor(self):
        counter = Counter(self.labels)
        infor ='\n'.join([label + ': ' + str(counter[label]) for label in self.data_labels])
        return infor

    def label2int(self, c: str):
        return

    def int2label(self, i: int):
        return

if __name__ == '__main__':

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

    dataset = TFDataset('./Dataset/train.csv', './Dataset/labels.json', img_transform=data_transforms)
    print(len(dataset))

    loader = DataLoader(dataset, min(8, len(dataset)), False)
    batch = next(iter(loader))
    print(batch[0])
    print(batch[1])