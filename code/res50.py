import torch
import torch.nn as nn
import torch.nn.functional as F



class IdentifyBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(IdentifyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 *planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 *planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4*planes)

        self.shortcut = nn.Sequential(nn.Conv2d(in_planes, 4*planes,kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(4*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50(nn.Module):

    def __init__(self, in_channels, num_classes=1000):
        super(ResNet50,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = nn.Sequential(ConvBlock(64,64),
                                    IdentifyBlock(256,64),
                                    IdentifyBlock(256,64),)

        self.layer2 = nn.Sequential(ConvBlock(256,128),
                                    IdentifyBlock(512,128),
                                    IdentifyBlock(512,128),
                                    IdentifyBlock(512,128))

        self.layer3 = nn.Sequential(ConvBlock(512,256),
                                    IdentifyBlock(1024,256),
                                    IdentifyBlock(1024,256),
                                    IdentifyBlock(1024,256),
                                    IdentifyBlock(1024,256),
                                    IdentifyBlock(1024,256))
        
        self.layer4 = nn.Sequential(ConvBlock(1024,512),
                                    IdentifyBlock(2048,512),
                                    IdentifyBlock(2048,512),)        
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x