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


class MaskDetectionDataset(Dataset):
    """
    MaskDetection image  ...
    """
    def __init__(self, csv_file, label_file, img_transform):
        """
        Args:
            csv_file: file contains data path and label
            label_file: file mapping label
            img_transform: img transform
        """
        data_labels = []
        f = open(label_file,)
        labels = json.load(f)
        print('label: ', labels)
        for label in labels:
            data_labels.append(labels[label])
        self.data_labels = data_labels

        df = pd.read_csv(csv_file)
        self.paths = df.iloc[:, 0]
        self.labels = df.iloc[:, 1]
        self.img_transform = img_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        # print('img_path: ', img_path)
        image = Image.open(img_path)
        # print('image: ', image)
        image = self.img_transform(image)
        
        label = self.label2int(self.labels[idx])
        return image, label

    def get_classweight(self):
        counter = Counter(self.labels)
        class_weights = torch.Tensor([(1.0 / counter[label]) for label in self.data_labels])
        return class_weights

    def get_infor(self):
        counter = Counter(self.labels)
        infor ='\n'.join([label + ': ' + str(counter[label]) for label in self.data_labels])
        return infor

    def label2int(self, c: str):
        return torch.tensor(self.data_labels.index(c))

    def int2label(self, i: int):
        return self.data_labels[int(i)]

if __name__ == '__main__':

    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

    dataset = MaskDetectionDataset('./Dataset/train.csv', './Dataset/labels.json', img_transform=data_transforms)
    print(len(dataset))

    loader = DataLoader(dataset, min(8, len(dataset)), False)
    batch = next(iter(loader))
    print(batch[0])
    print(batch[1])