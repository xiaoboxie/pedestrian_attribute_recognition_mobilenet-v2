#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:27:37 2018

@author: bobby
"""

with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
imgpathset = [i.split()[0] for i in dataset]
labelset = [i.split()[1:] for i in dataset]
#remove several attribute
newlabelset = [i.remove() for i in labelset]