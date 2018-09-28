#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:59:12 2018

@author: bobby
"""

from PIL import Image


def hot2seq(alist):
    alist = [int(i) for i in alist]
    blist = []
    for i,j in enumerate(alist):
        if j:
            blist.append(i)
    return blist

def getshape(imgpath):
    img = Image.open(imgpath)
    return img.size

with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
    
dataset = [i.replace('./','/home/bobby/work/dataset/bussiness100/') for i in dataset]

newdataset = []
for i in dataset:
    name = i.split()[0]
    label = i.split()[1:]
    newlabel = hot2seq(label)
    
    shape = getshape(name)
    box = '0,0,{},{},'.format(shape[0],shape[1])
    
    newlabels = ''
    for j in newlabel:
        newlabels += (' '+box+str(j))
    newdataset.append(name+newlabels)

with open('/home/bobby/work/dataset/bussiness100/train_yolo.txt', 'w') as f:
    dataset = f.writelines([i+'\n' for i in newdataset])