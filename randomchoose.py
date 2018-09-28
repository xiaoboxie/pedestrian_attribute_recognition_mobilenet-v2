#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:27:07 2018

@author: bobby
"""
import random
newdataset = []
with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
for i in dataset:
    if random.choice([1,1,1,1,0]):
        newdataset.append(i)
with open('/home/bobby/work/dataset/bussiness100/train%80.txt', 'w') as f:
    f.writelines(newdataset)