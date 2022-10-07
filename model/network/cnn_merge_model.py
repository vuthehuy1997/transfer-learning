import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from . import fe
class CNNMergeModel(nn.Module):
    def __init__(self, model1, model2, number_class=3, drop_p=0.3):
        super(CNNMergeModel, self).__init__()
        self.backbone1 = model1.backbone
        self.backbone2 = model2.backbone
        self.n_features =  model1.n_features + model2.n_features
        self.fc_cnn = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(512, number_class)
        )

    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """

        with torch.no_grad():
            x1 = self.backbone1(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
            x1 = torch.flatten(x1, start_dim=1) # [batch, num_features]
            x2 = self.backbone2(images) # [batch, num_features, H', W'] = [batch, 2048, 1, 1]
            x2 = torch.flatten(x2, start_dim=1) # [batch, num_features]
        x = torch.cat((x1,x2), 1)
        x = self.fc_cnn(x)
        return x
