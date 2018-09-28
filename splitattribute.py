#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:22:57 2018

@author: bobby
"""
import re
with open('/home/bobby/work/code/bussiness100/each_attribute_count_orig.txt') as f:
    attcountfile = f.readlines()
    attcount = [int(i.split()[0]) for i in attcountfile]
for i in range(len(attcount)):
    for j in range(len(attcount)-i-1):
        if attcount[j]<attcount[j+1]:
            attcount[j+1],attcount[j]=attcount[j],attcount[j+1]
            attcountfile[j+1],attcountfile[j]=attcountfile[j],attcountfile[j+1]
print(attcount)
attcount1 = attcountfile[0:29]
attcount2 = attcountfile[29:-45]
attcount3 = attcountfile[-45:]

compiler = re.compile('(?<=[\u4e00-\u9fa5])\d+')
attcount1_index = []
attcount2_index = []
attcount3_index = []
for ac,aci in zip([attcount1,attcount2,attcount3],[attcount1_index,attcount2_index,attcount3_index]):
    for i in ac:
        tmp = int(re.findall(compiler, i)[0])
        aci.append(tmp)
    
with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
    split1 = [i.split()[0]+' '+' '.join([i.split()[1:][j] for j in attcount1_index]) for i in dataset]
    split2 = [i.split()[0]+' '+' '.join([i.split()[1:][j] for j in attcount2_index]) for i in dataset]
    split3 = [i.split()[0]+' '+' '.join([i.split()[1:][j] for j in attcount3_index]) for i in dataset]        

name = ['splitH','splitM','splitL']
for i,j in zip(name,[split1,split2,split3]):
    with open('/home/bobby/work/dataset/bussiness100/{}.txt'.format(i), 'w') as f:
        f.writelines([tmp+'\n' for tmp in j if '1' in tmp.split()[1:]])