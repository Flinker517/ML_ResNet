from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import trange
from PIL import Image
import argparse
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from apex import amp

def getAUC(y_true, y_score, task):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param task: the task of current dataset

    '''
    if task == 'binary-class':
        y_score = y_score[:,-1]
        return roc_auc_score(y_true, y_score)
    elif task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        return auc / y_score.shape[1]
    else:
        auc = 0
        zero = np.zeros_like(y_true)
        one = np.ones_like(y_true)
        for i in range(y_score.shape[1]):
            y_true_binary = np.where(y_true == i, one, zero)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        return auc / y_score.shape[1]


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks

    '''
    if task == 'multi-label, binary-class':
        zero = np.zeros_like(y_score)
        one = np.ones_like(y_score)
        y_pre = np.where(y_score < threshold, zero, one)
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        return acc / y_true.shape[1]
    elif task == 'binary-class':
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = (y_score[i][-1] > threshold)
        return accuracy_score(y_true, y_pre)
    else:
        y_pre = np.zeros_like(y_true)
        for i in range(y_score.shape[0]):
            y_pre[i] = np.argmax(y_score[i])
        return accuracy_score(y_true, y_pre)


def save_results(y_true, y_score, outputpath):
    '''Save ground truth and scores
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param outputpath: path to save the result csv

    '''
    idx = []

    idx.append('id')

    for i in range(y_true.shape[1]):
        idx.append('true_%s' % (i))
    for i in range(y_score.shape[1]):
        idx.append('score_%s' % (i))

    df = pd.DataFrame(columns=idx)
    for id in range(y_score.shape[0]):
        dic = {}
        dic['id'] = id
        for i in range(y_true.shape[1]):
            dic['true_%s' % (i)] = y_true[id][i]
        for i in range(y_score.shape[1]):
            dic['score_%s' % (i)] = y_score[id][i]

        df_insert = pd.DataFrame(dic, index = [0])
        df = df.append(df_insert, ignore_index=True)

    df.to_csv(outputpath, sep=',', index=False, header=True, encoding="utf_8_sig")

class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        self.info = INFO[self.flag]
        self.root = root

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.
        '''
        _repr_indent = 4
        head = "Dataset " + self.__class__.__name__

        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Split: {}".format(self.split))
        body.append("Task: {}".format(self.info["task"]))
        body.append("Number of channels: {}".format(self.info["n_channels"]))
        body.append("Meaning of labels: {}".format(self.info["label"]))
        body.append("Number of samples: {}".format(self.info["n_samples"]))
        body.append("Description: {}".format(self.info["description"]))
        body.append("License: {}".format(self.info["license"]))

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               'https://github.com/MedMNIST/MedMNIST')


class PathMNIST(MedMNIST):
    flag = "pathmnist"


class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNISTAxial(MedMNIST):
    flag = "organmnist_axial"


class OrganMNISTCoronal(MedMNIST):
    flag = "organmnist_coronal"


class OrganMNISTSagittal(MedMNIST):
    flag = "organmnist_sagittal"


INFO = {
    "pathmnist": {
        "description":
        "PathMNIST: A dataset based on a prior study for predicting survival from colorectal cancer histology slides, which provides a dataset NCT-CRC-HE-100K of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset CRC-VAL-HE-7K of 7,180 image patches from a different clinical center. 9 types of tissues are involved, resulting a multi-class classification task. We resize the source images of 3 x 224 x 224 into 3 x 28 x 28, and split NCT-CRC-HE-100K into training and valiation set with a ratio of 9:1.",
        "url":
        "https://zenodo.org/record/4269852/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 89996,
            "val": 10004,
            "test": 7180
        },
        "license": "CC BY 4.0"
    },
    "chestmnist": {
        "description":
        "ChestMNIST: A dataset based on NIH-ChestXray14 dataset, comprising 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined 14 disease image labels, which could be formulized as multi-label binary classification task. We use the official data split, and resize the source images of 1 x 1024 x 1024 into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/4269852/files/chestmnist.npz?download=1",
        "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        "task": "multi-label, binary-class",
        "label": {
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "pneumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 78468,
            "val": 11219,
            "test": 22433
        },
        "license": "CC0 1.0"
    },
    "dermamnist": {
        "description":
        "DermaMNIST: A dataset based on HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images labeled as 7 different categories, as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3 x 600 x 450 are resized into 3 x 28 x 28.",
        "url":
        "https://zenodo.org/record/4269852/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 7007,
            "val": 1003,
            "test": 2005
        },
        "license": "CC BY-NC 4.0"
    },
    "octmnist": {
        "description":
        "OCTMNIST: A dataset based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. 4 types are involved, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-1,536) x (277-512). We center-crop the images and resize them into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/4269852/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 97477,
            "val": 10832,
            "test": 1000
        },
        "license": "CC BY 4.0"
    },
    "pneumoniamnist": {
        "description":
        "PneumoniaMNIST: A dataset based on a prior dataset of 5,856 pediatric chest X-ray images. The task is binary-class classification of pneumonia and normal. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-2,916) x (127-2,713). We center-crop the images and resize them into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/4269852/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },
        "license": "CC BY 4.0"
    },
    "retinamnist": {
        "description":
        "RetinaMNIST: A dataset based on DeepDRiD, a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as test set. The source images of 3 x 1,736 x 1,824 are center-cropped and resized into 3 x 28 x 28",
        "url":
        "https://zenodo.org/record/4269852/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "task": "ordinal regression",
        "label": {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 1080,
            "val": 120,
            "test": 400
        },
        "license": "CC BY 4.0"
    },
    "breastmnist": {
        "description":
        "BreastMNIST: A dataset based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign and malignant. As we use low-resolution images, we simplify the task into binary classification by combing normal and benign as negative, and classify them against malignant as positive. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1 x 500 x 500 are resized into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/4269852/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "task": "binary-class",
        "label": {
            "0": "malignant",
            "1": "normal, benign"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 546,
            "val": 78,
            "test": 156
        },
        "license": "CC BY 4.0"
    },
    "organmnist_axial": {
        "description":
        "OrganMNIST_Axial: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/4269852/files/organmnist_axial.npz?download=1",
        "MD5": "866b832ed4eeba67bfb9edee1d5544e6",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 34581,
            "val": 6491,
            "test": 17778
        },
        "license": "CC BY 4.0"
    },
    "organmnist_coronal": {
        "description":
        "OrganMNIST_Coronal: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/4269852/files/organmnist_coronal.npz?download=1",
        "MD5": "0afa5834fb105f7705a7d93372119a21",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13000,
            "val": 2392,
            "test": 8268
        },
        "license": "CC BY 4.0"
    },
    "organmnist_sagittal": {
        "description":
        "OrganMNIST_Sagittal: A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/4269852/files/organmnist_sagittal.npz?download=1",
        "MD5": "e5c39f1af030238290b9557d9503af9d",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13940,
            "val": 2452,
            "test": 8829
        },
        "license": "CC BY 4.0"
    }
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels, num_classes=1000):
        super().__init__()
        
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  
   
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)           # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # 112x112

        x = self.layer1(x)          # 56x56
        x = self.layer2(x)          # 28x28
        x = self.layer3(x)          # 14x14
        x = self.layer4(x)          # 7x7

        x = self.avgpool(x)         # 1x1
        x = torch.flatten(x, 1)     # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc(x)

        return x
        
        
def ResNet18(in_channels, num_classes):
    layers=[2, 2, 2, 2]
    model = ResNet(BasicBlock, layers, in_channels, num_classes)
    return model

def train(model, optimizer, criterion, train_loader, device, task):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.cuda(non_blocking=True))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).cuda(non_blocking=True)
            loss = criterion(outputs, targets).cuda(non_blocking=True)
        else:
            targets = targets.squeeze().long().cuda(non_blocking=True)
            loss = criterion(outputs, targets).cuda(non_blocking=True)

        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

    model.eval()
    y_true = torch.tensor([]).cuda(non_blocking=True)
    y_score = torch.tensor([]).cuda(non_blocking=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.cuda(non_blocking=True))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).cuda(non_blocking=True)
                m = nn.Sigmoid()
                outputs = m(outputs).cuda(non_blocking=True)
            else:
                targets = targets.squeeze().long().cuda(non_blocking=True)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).cuda(non_blocking=True)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)
    
def test(model, split, data_loader, flag, task, output_root=None):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param flag: subset name
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).cuda(non_blocking=True)
    y_score = torch.tensor([]).cuda(non_blocking=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.cuda(non_blocking=True))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).cuda(non_blocking=True)
                m = nn.Sigmoid()
                outputs = m(outputs).cuda(non_blocking=True)
            else:
                targets = targets.squeeze().long().cuda(non_blocking=True)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).cuda(non_blocking=True)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)



if __name__ == '__main__':

    input_root = '../data'
    output_root = './output'

    parser = argparse.ArgumentParser(
        description='RUN ResNet18.')
    parser.add_argument('--data_name',
                        default='pathmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    
    data_name = args.data_name.lower()

    name_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    
    DataClass = name_to_class[data_name]
    info = INFO[data_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    start_epoch = 0
    end_epoch = 100
    lr = 0.001
    batch_size = 128
    val_auc_list = []
    dir_path = os.path.join(".", '%s_checkpoints' % (data_name))
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = DataClass(root=input_root,
                                split='train',
                                transform=transform,
                                download=False)
                                
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              num_workers=4,sampler=train_sampler)
    val_dataset = DataClass(root=input_root,
                                  split='val',
                                  transform=transform,
                                  download=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=input_root,
                                   split='test',
                                   transform=transform,
                                   download=False)
                                  
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
                                             
    print('==> Building and training model...')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet18(in_channels=n_channels, num_classes=n_classes)#.to(device)

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model = DistributedDataParallel(model)#, device_ids=[args.local_rank]
    model, optimizer = amp.initialize(model, optimizer)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    
    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch)
    
    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model,
         'train',
         train_loader,
         data_name,
         task,
         output_root=output_root)
    test(model, 'val', val_loader, data_name, task, output_root=output_root)
    test(model,
         'test',
         test_loader,
         data_name,
         task,
         output_root=output_root)
    
    