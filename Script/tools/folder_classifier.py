#!/usr/bin/env python
# coding: utf-8

import shutil
import os
import csv
from os import listdir
from os.path import join, isdir

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

path = 'D:\\AI_Mango\\data\\C1-P1_Train'
img_root = path+'\\'

if isdir(img_root+'A'):
    pass
else:
    os.mkdir(img_root+'A')
    os.mkdir(img_root+'B')
    os.mkdir(img_root+'C')
    
test_csv = ReadCSV('D:\\AI_Mango\\data\\C1-P1_Train.csv')
test_label = test_csv.read()

img = listdir(path)
print(len(test_label), len(img))

for i in range(len(test_label)):
    fullpath = join(img_root, test_label[i][0])
    if test_label[i][1] == 'A':
        shutil.copyfile(fullpath, img_root+'A\\'+test_label[i][0])
    elif test_label[i][1] == 'B':
        shutil.copyfile(fullpath, img_root+'B\\'+test_label[i][0])
    elif test_label[i][1] == 'C':
        shutil.copyfile(fullpath, img_root+'C\\'+test_label[i][0])
print('done!')

