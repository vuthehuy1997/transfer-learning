import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

#-------------v1
class CNNModel(nn.Module):
    def __init__(self, number_class=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, number_class)

    def forward(self, images):
        """
        Args:
            images: a tensor of dimension [batch, 3, height, width]
        Return:
            outputs: a tensor of dimension [batch, num_classes]
        """
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#-------------v2
# class CNNModel(nn.Module):
#     def __init__(self, number_class=2):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, number_class)

#     def forward(self, image):
#         x = self.pool(F.relu(self.conv1(image)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x