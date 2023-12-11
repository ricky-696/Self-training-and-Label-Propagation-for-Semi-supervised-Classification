# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:11:41 2021

@author: b3171154
"""
import torch.nn as nn 
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch
from torch.nn import functional as F
from torchvision import models

class gray_resnet18(nn.Module):
    def __init__(self, num_classes):
        super(gray_resnet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


class FC(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FC, self).__init__()
        
        self.fc1 = nn.Linear(num_classes * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(20, 16) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MNISTResNet(ResNet):
    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.reshape(-1,shape)
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        self.resnet = models.resnet18(pretrained=True)
 
    def forward(self, x):
 
        x = self.conv(x)
        x = self.resnet(x)

        return x

def MNISTVGG():

    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        Flatten(),
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.Linear(in_features=4096, out_features=10, bias=True)
    )
    
    return model
