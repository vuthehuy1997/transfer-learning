import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, feature_extract=True, pretrained=True, number_class=3, drop_p=0.3):
        super(CNNModel, self).__init__()
        mobilenet = torchvision.models.mobilenet_v2(pretrained = pretrained)
        print('last: ', mobilenet.last_channel)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        if feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc_cnn = nn.Sequential(
            nn.Linear(mobilenet.last_channel, 512),
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
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone(images) #output [batch, num_features, H', W'] = [batch, 2048, 1, 1]
            x = torch.flatten(x, start_dim=1) # [batch, num_features]
        # print(x.shape)
        x = self.fc_cnn(x)
        return x