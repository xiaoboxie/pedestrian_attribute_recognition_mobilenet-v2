#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:31:06 2018

@author: bobby
"""

with open('/home/bobby/work/dataset/bussiness100/val_new.txt') as f:
    dataset = f.readlines()
    test = [j.split()[0]+' '+' '.join(j.split()[1:][:48]+j.split()[1:][57:]) for j in dataset]
    
with open('/home/bobby/work/dataset/bussiness100/val_test.txt', 'w') as f:
    f.writelines([i+'\n' for i in test if '1' in i.split()[1:]])