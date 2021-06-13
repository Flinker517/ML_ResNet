from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import trange
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import res50

class chestmnist(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        npz_file = np.load(self.root)

        self.split = split
        self.transform = transform

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

        return img, target
    def __len__(self):
        return self.img.shape[0]


def getAUC(y_true, y_score):
    auc = 0
    for i in range(y_score.shape[1]):
        label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
        auc += label_auc
    return auc / y_score.shape[1]

def getACC(y_true, y_score, threshold=0.5):
    zero = np.zeros_like(y_score)
    one = np.ones_like(y_score)
    y_pre = np.where(y_score < threshold, zero, one)
    acc = 0
    for label in range(y_true.shape[1]):
        label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
        acc += label_acc
    return acc / y_true.shape[1]

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        # if task == 'multi-label, binary-class':
        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets).to(device)
        loss.backward()
        optimizer.step()

def val(model, val_loader, device, val_auc_list, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            # if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        val_auc_list.append(auc)

    state = {'net': model.state_dict(), 'auc': auc, 'epoch': epoch}
    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)
    
def test(model, data_loader):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            # if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            m = nn.Sigmoid()
            outputs = m(outputs).to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        acc = getACC(y_true, y_score)
        print('AUC: %.5f ACC: %.5f' % (auc, acc))


input_root = '../data/chestmnist.npz'
output_root = '../output1/chestmnist'
if not os.path.exists(output_root):
	os.mkdir(output_root)

start_epoch = 0
end_epoch = 40
batch_size = 128
val_auc_list = []

print('Load data...')
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = chestmnist(root=input_root, split='train', transform=transform)                     
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = chestmnist(root=input_root, split='val', transform=transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = chestmnist(root=input_root, split='test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                                             
print('train model...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = res50.ResNet50(in_channels=1, num_classes=14).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

for epoch in trange(start_epoch, end_epoch):
    train(model, optimizer, criterion, train_loader, device)
    val(model, val_loader, device, val_auc_list, output_root, epoch)
    
auc_list = np.array(val_auc_list)
index = auc_list.argmax()
print('epoch %s is the best model' % (index))

print('Test model...')
restore_model_path = os.path.join(output_root, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
model.load_state_dict(torch.load(restore_model_path)['net'])
test(model, train_loader)
test(model, val_loader)
test(model, test_loader)
    
    