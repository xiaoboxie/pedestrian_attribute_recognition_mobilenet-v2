#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:02:10 2018

@author: bobby
"""

path = '/home/bobby/work/dataset/bussiness100/train_new (复件).txt'
with open(path) as f:
    dataset = f.readlines()
    labelset = [[int(j) for j in i.split()[1:]] for i in dataset]
    count_list = []
    for i in range(len(labelset[0])):
        count = 0
        for j in labelset:
            count += j[i]
        count_list.append(count)
savepath = path[0:-4]+'_count'+path[-4:]

with open(savepath) as f:
    f.writelines([i+'\n' for i in count_list])