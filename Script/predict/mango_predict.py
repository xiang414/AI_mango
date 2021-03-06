#!/usr/bin/env python
# coding: utf-8
'''
torch : 1.5.0
torchvision : 0.6.0
numpy : 1.18.1
PIL : 7.1.2
'''
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import csv
from collections import OrderedDict
from torch import nn
from PIL import Image, ImageEnhance
from os import listdir
from os.path import join

#----------------------alexnet----------------------
model = torchvision.models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
model.load_state_dict(torch.load(r"..\model\alex.pth"))
print(model)

# for value in model["classifier.fc3.weight"].item():
    # print(value.size())

device = torch.device('cuda')
model = model.to(device)

def process_image(image):
    #調整圖片大小
    img = Image.open(image).convert('RGB')
    #enh_col = ImageEnhance.Color(img)
    #img = enh_col.enhance(1.5)
    #enh_con = ImageEnhance.Contrast(img_colored)
    #pic = enh_con.enhance(1.5))
    
    img = transforms.Compose([
                              transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                             ])(img)
    return img

def predict(image_path, model):
    img = process_image(image_path)
    img = img.unsqueeze(0)   #將圖片多增加一維
    with torch.no_grad():
        model.eval()
        result = model(img.cuda())
        _, index = torch.max(result, 1)
        index = str(index.item())
    return index

pic_root = r"..\..\dataset\Testing Data\sample_image"
file = listdir(pic_root)

with open(r"..\..\dataset\Testing Data\output.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for pic in file:
        pic_dir = join(pic_root, pic)
        classes = predict(pic_dir, model)
        if classes == '0':
            prediction = 'A'
        elif classes == '1':
            prediction = 'B'
        elif classes == '2':
            prediction = 'C'
        print(pic +'      '+ prediction)
        writer.writerow([pic, prediction])
print('Output Successfully')
