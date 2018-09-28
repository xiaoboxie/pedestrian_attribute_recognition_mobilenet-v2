#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:35:42 2018

@author: bobby
"""
filepath = '/home/bobby/work/code/bussiness100/prediction_raptestlist_modified.txt'
#'/home/bobby/work/code/bussiness100/prediction_raptestlist_modified.txt'

with open(filepath) as f:
    a = f.readlines()
    b = [i.split(' ')[0]+''.join([' '+str(j) for j in eval(i.split('png ')[1:][0])]) for i in a]

with open('temp2.txt','w') as f:
    f.writelines([i+'\n' for i in b])