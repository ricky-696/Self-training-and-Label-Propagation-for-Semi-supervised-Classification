# -*- coding: utf-8 -*-
import torch
import torch.nn as nn 
from torchvision import models
from torch.nn import functional as F

class gray_resnet18(nn.Module):
    def __init__(self, num_classes):
        super(gray_resnet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


class resnet18(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super(resnet18, self).__init__()
        if pretrain:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


class resnet50(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super(resnet50, self).__init__()
        if pretrain:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


class DenseNet169_BC(nn.Module):
    def __init__(self, num_classes=2, pretrain=False):
        super(DenseNet169_BC, self).__init__()
        if pretrain:
            self.densenet = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        else:
            self.densenet = models.densenet169()
            
        self.densenet.classifier = nn.Linear(1664, num_classes)

    def forward(self, x):
        return self.densenet(x)
    
    
class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, pretrain=False):
        super(DenseNet121, self).__init__()
        if pretrain:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        else:
            self.densenet = models.densenet121()
            
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.densenet(x)


class FC_3layer(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FC_3layer, self).__init__()
        
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


class FC_2layer(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FC_2layer, self).__init__()
        
        self.fc1 = nn.Linear(num_classes * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        
        return x


class FC_1layer(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(FC_1layer, self).__init__()
        
        self.fc1 = nn.Linear(num_classes * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
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
        out = F.softmax(out, dim=1)

        return out


class Avg_Label(nn.Module):
    def __init__(self, num_classes):
        super(Avg_Label, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x1, x2 = x.split(self.num_classes, dim=1)
        avg_probs = (x1 + x2) / 2.0
        avg_probs = F.softmax(avg_probs, dim=1)

        return avg_probs
    

class Same_Label(nn.Module):
    def __init__(self, num_classes, device):
        super(Same_Label, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, x):
        x1, x2 = x.split(self.num_classes, dim=1)

        argmax_x1 = torch.argmax(x1, dim=1)
        argmax_x2 = torch.argmax(x2, dim=1)

        condition = argmax_x1 == argmax_x2

        # 使用torch.where选择输出值
        pseudo_labels = torch.where(
            condition.unsqueeze(1),
            F.softmax((x1 + x2) / 2.0, dim=1),
            torch.zeros((x.size(0), self.num_classes)).to(self.device)
        )

        return pseudo_labels


if __name__ == '__main__':
    model = DenseNet121().to('cpu')
    
    for name, param in model.named_parameters():
        print(name, param.requires_grad)