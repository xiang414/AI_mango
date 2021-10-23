#!/usr/bin/env python
# coding: utf-8
'''
torch : 1.5.0
torchvision : 0.6.0
numpy : 1.18.1
'''
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
from matplotlib import pyplot as plt

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
    #print("correct number:", correct, "  ", "total number:", total)
    return correct/total, test_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
#----------------------alexnet----------------------
BEST_MODEL_PATH = r"..\model\res18.pth"
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 3)
for param in model.parameters():
    param.require_grad = False
print(model)

weight_p, bias_p = [], []
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]

# classifier_layers_params = list(map(id, model.classifier.parameters()))
# base_params = filter(lambda p:id(p) not in classifier_layers_params, model.parameters())
fc_layers_params = list(map(id, model.fc.parameters()))
base_params = filter(lambda p:id(p) not in fc_layers_params, model.parameters())

if __name__ == '__main__':
    run()
#------------------------------datasets------------------------------
    train_dataset = datasets.ImageFolder(
        r"..\..\dataset\ImageFolder Data\all_data\train",
        transforms.Compose([
            #transforms.Resize((244, 244)),
            #transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

    test_dataset = datasets.ImageFolder(
        r"..\..\dataset\ImageFolder Data\all_data\test",
        transforms.Compose([
            #transforms.Resize((244, 244)),
            #transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    print(len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=True, num_workers=0)
#------------------------------load model to GPU------------------------------
    device = torch.device('cuda')
    model = model.to(device)
#------------------------------start training------------------------------
    NUM_EPOCHS = 150
    best_accuracy = 0.0
    history = []
    early_stopping = EarlyStopping(patience=7, verbose=True, path=BEST_MODEL_PATH)

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.5)
    '''optimizer = optim.SGD([{'params': model.classifier.parameters(), 'lr':1e-3, 'weight_decay':1e-4},
                           {'params': base_params, 'lr':1e-4}],
                           momentum=0.9)'''
    optimizer = optim.SGD([{'params': weight_p, 'weight_decay':5e-2},
                           {'params': bias_p, 'weight_decay':0}], 
                           lr=1e-3, 
                           momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    tune_lr = optim.lr_scheduler.ReduceLROnPlateau( #使用動態學習率
                                                   optimizer,
                                                   mode='max',
                                                   factor=0.5,
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
            optimizer.zero_grad()   #將模型的參數梯度初始化為0
            outputs = model(images) #前向傳播計算預測值
            loss = nn.CrossEntropyLoss()
            loss = loss(outputs, labels)
            train_loss += float(loss.item()*images.size(0))
            loss.backward()   #反向傳播計算梯度
            optimizer.step()  #更新所有參數
            _, pred = torch.max(outputs, 1)
            num_correct = (pred==labels).sum()
            train_acc_count += float(num_correct.item())
        train_loss = train_loss/len(train_loader.dataset)
        train_accuracy = float(train_acc_count) / float(len(train_dataset))

        test_accuracy, test_loss = accuracy_test(model, test_loader)
        test_loss = test_loss/len(test_loader.dataset)
        history.append([train_accuracy, test_accuracy, train_loss, test_loss])
        tune_lr.step(test_accuracy)
        
        print('Epoch: %d: Train Acc: %f, Valid Acc: %f \n\t  Train Loss: %f, Valid Loss: %f' % 
                (epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            break
        
        if train_accuracy >= 0.7:
            if test_accuracy >= best_accuracy:
                early_stopping.counter = 0
                print('Validation accuracy increase ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    best_accuracy, test_accuracy))
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                best_accuracy = test_accuracy
        print('----------------------------------------------------------')
    print('training complete')
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
    ax2.set_ylim(0, 1)
    plt.show()
