#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image, ImageEnhance

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ReadCSV:
    def __init__(self, path):
        self.path = path
        self.rows = None
        self.all_list = []
        self.label = []

    def read(self):
        with open(self.path) as csvfile:
            self.rows = csv.reader(csvfile)
            for row in self.rows:
                self.all_list.append(row)
        for i in range(1,len(self.all_list)):
            self.label.append(self.all_list[i])
        return self.label

class CustomDataset(Dataset):
    def __init__(self, root, label, data):
        self.imgs = label
        self.root = root
        self.data = data
    
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(self.root+path).convert('RGB')
        '''enh_col = ImageEnhance.Color(img)
        img_colored = enh_col.enhance(1.5)
        enh_con = ImageEnhance.Contrast(img_colored)
        img = enh_con.enhance(1.5)'''
        if self.data == 'train':
            img = transforms.Compose([
                                      #transforms.Resize((264, 264)),
                                      #transforms.CenterCrop(224),
                                      transforms.Resize((224, 224)),
                                      transforms.RandomRotation(15),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])(img)
        else:
            img = transforms.Compose([
                                      #transforms.Resize((264, 264)),
                                      #transforms.CenterCrop(224),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    ])(img)
        if label == 'A':
            label = 0
        elif label == 'B':
            label = 1
        else:
            label = 2
        return img, label
    
    def __len__(self):
        return len(self.imgs)

def run():
    torch.multiprocessing.freeze_support()

def accuracy_test(model, dataloader):
    correct = 0.0
    total = 0.0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():  #使用驗證集時關閉梯度計算
        for images, labels in iter(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            test_loss += float(loss.item()*images.size(0))
            _, predicted = torch.max(outputs, 1) #torch.max返回輸出結果中，按dim=1排列的每一列最大數據及他的索引，在這裡只取用索引
            total += labels.size(0)   #tensor的列大小,total=len(test_dataset)
            correct += (predicted==labels).sum().item()  #將預測及標籤兩相同大小張量逐一比較各相同元素的個數
            
    return correct/total, test_loss

train_root = r"..\..\dataset\Custom Data\all_train\\"
train_csv = r"..\..\dataset\Custom Data\all_train.csv"
test_root = r"..\..\dataset\Custom Data\train\\"
test_csv = r"..\..\dataset\Custom Data\train.csv"

train_csv = ReadCSV(train_csv)
train_label = train_csv.read()

test_csv = ReadCSV(test_csv)
test_label = test_csv.read()
#----------------------alexnet----------------------
BEST_MODEL_PATH = r"..\model\resnet18.pth"
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 3)
for param in model.parameters():
    param.require_grad = False
print(model)

weight_p, bias_p = [],[]
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

fc_layers_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p:id(p) not in fc_layers_params, model.parameters())

if __name__ == '__main__':
    run()
    train_dataset = CustomDataset(train_root, train_label, 'train')
    test_dataset = CustomDataset(test_root, test_label, 'test')
    print("testing dataset count: ", len(test_label))

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=True, num_workers=0)

    img, label = test_loader.__iter__().__next__()
    print("check label: ", label)

    device = torch.device('cuda')
    model = model.to(device)

    NUM_EPOCHS = 200
    best_accuracy = 0.0
    history = []
    early_stopping = EarlyStopping(patience=7, verbose=True, path=BEST_MODEL_PATH)

    optimizer = optim.SGD([{'params': model.fc.parameters(), 'lr':1e-3},
                           {'params': base_params, 'lr':5e-5}],
                           momentum=0.9,
                           weight_decay=1e-4)
    # optimizer = optim.SGD([{'params': weight_p, 'weight_decay':1e-2},
                           # {'params': bias_p, 'weight_decay':0, 'lr':1e-5}], 
                           # lr=1e-3, 
                           # momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    #optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    tune_lr = optim.lr_scheduler.ReduceLROnPlateau( #使用動態學習率
                                                   optimizer,
                                                   mode='max',
                                                   factor=0.7,
                                                   patience=5,
                                                   verbose=True,
                                                   threshold=0.0001,
                                                   threshold_mode='rel',
                                                   cooldown=0,
                                                   min_lr=0,
                                                   eps=1e-08
                                                   )
    for epoch in range(NUM_EPOCHS):
    
        model.train()
        train_acc_count = 0.0
        train_loss = 0.0

        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            num_correct = (pred==labels).sum()
            train_acc_count += float(num_correct.item())
            train_loss += float(loss.item()*images.size(0))
        train_accuracy = float(train_acc_count) / float(len(train_dataset))
        train_loss = train_loss/float(len(train_dataset))

        test_accuracy, test_loss = accuracy_test(model, test_loader)
        test_loss = test_loss/len(test_loader.dataset)
        history.append([train_accuracy, test_accuracy, train_loss, test_loss])
        tune_lr.step(train_accuracy)
    
        print('Epoch: %d: Train Acc: %f, Valid Acc: %f \n\t  Train Loss: %f, Valid Loss: %f' % 
            (epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
    
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if train_accuracy >= 0.7:
            if test_accuracy >= best_accuracy:
                early_stopping.counter = 0
                print('Validation accuracy increase ({:.4f} --> {:.4f}).  Saving model ...'.format(
                        best_accuracy, test_accuracy))
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                best_accuracy = test_accuracy
        print('----------------------------------------------------')
    history = np.array(history)
    fig = plt.figure(figsize=[12.8,4.8])
    ax1 = plt.subplot(121)
    ax1.plot(history[:,0:2])
    ax1.legend(['Train Acc', 'Valid Acc'], loc='best')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax2 = plt.subplot(122)
    ax2.plot(history[:,2:4])
    ax2.legend(['Train Loss', 'Valid Loss'], loc='best')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 1.2)
    plt.show()
    print('training complete')