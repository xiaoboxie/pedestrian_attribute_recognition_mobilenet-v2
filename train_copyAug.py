#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 19:37:41 2018

@author: bobby
"""

import random
import copy

with open('/home/bobby/work/code/bussiness100/each_attribute_count.txt') as f:
    att_count = f.readlines()
    att_count = [int(i.split()[0]) for i in att_count]
with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
    dataset_copy = copy.deepcopy(dataset)
    for i in range(len(att_count)):
        if att_count[i]<3000:
            rarelist = [j if j.split()[1:][i]=='1' else None for j in dataset]
            while None in rarelist:
                rarelist.remove(None)
            assert att_count[i]==len(rarelist)
            tmp = 3000//att_count[i]
            rarelist = rarelist*tmp
            dataset_copy.extend(rarelist)
            
random.shuffle(dataset_copy)

with open('/home/bobby/work/dataset/bussiness100/train_copyAug.txt','w') as f:
    f.writelines(dataset_copy)            