#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:50:21 2018

@author: bobby
"""
import copy
def classtransfer(bus_label, rap_label):
    '''
        a = score[0:8]
        b = score[8:24]
        c = score[24:33]
        d = score[33:39]
        e = score[39:43]
        f = score[43:48]
        g = score[48:57]
        h = score[57:73]
        i = score[73:89]
        j = score[89:96]
        k = score[96:101]
        l = score[101:108]
        m = score[108:111]
        n = score[111:118]
        o = score[118:130]
    '''
    if rap_label==[1]*92:
        return bus_label
    if rap_label[83]==1:
        bus_label[8:24]=[0]*16
        bus_label[23]=1
    if rap_label[84]==1:
        bus_label[8:24]=[0]*16
        bus_label[14]=1
    if rap_label[85]==1:
        bus_label[8:24]=[0]*16
        bus_label[13]=1
    if rap_label[86]==1:
        bus_label[8:24]=[0]*16
        bus_label[17]=1
    if rap_label[87]==1:
        bus_label[8:24]=[0]*16
        bus_label[18]=1
    if rap_label[88]==1:
        bus_label[8:24]=[0]*16
        bus_label[21]=1
    if rap_label[89]==1:
        bus_label[8:24]=[0]*16
        bus_label[22]=1
    
    if rap_label[14]==1:
        bus_label[24:33]=[0]*9
        bus_label[26]=1
    if rap_label[13]==1:
        bus_label[24:33]=[0]*9
        bus_label[31]=1
    if rap_label[12]==1:
        bus_label[24:33]=[0]*9
        bus_label[28]=1
        
    if rap_label[1]==1:
        bus_label[33:39]=[0]*6
        bus_label[36]=1
    if rap_label[2]==1:
        bus_label[33:39]=[0]*6
        bus_label[38]=1
    if rap_label[3]==1:
        bus_label[33:39]=[0]*6
        bus_label[34]=1
        
    if rap_label[18]==1:
        bus_label[48:57]=[0]*9
        bus_label[46]=1
    if rap_label[23]==1:
        bus_label[48:57]=[0]*9
        bus_label[46]=1
        
    if rap_label[63]==1:
        bus_label[57:73]=[0]*16
        bus_label[72]=1
    if rap_label[64]==1:
        bus_label[57:73]=[0]*16
        bus_label[63]=1
    if rap_label[65]==1:
        bus_label[57:73]=[0]*16
        bus_label[62]=1
    if rap_label[66]==1:
        bus_label[57:73]=[0]*16
        bus_label[66]=1
    if rap_label[67]==1:
        bus_label[57:73]=[0]*16
        bus_label[67]=1
    if rap_label[68]==1:
        bus_label[57:73]=[0]*16
        bus_label[70]=1
    if rap_label[69]==1:
        bus_label[57:73]=[0]*16
        bus_label[71]=1
    if rap_label[70]==1:
        bus_label[57:73]=[0]*16
        bus_label[59]=1
    if rap_label[71]==1:
        bus_label[57:73]=[0]*16
        bus_label[65]=1
    if rap_label[72]==1:
        bus_label[57:73]=[0]*16
        bus_label[64]=1
    if rap_label[73]==1:
        bus_label[57:73]=[0]*16
        bus_label[60]=1
    if rap_label[74]==1:
        bus_label[57:73]=[0]*16
        bus_label[69]=1
        
    if rap_label[75]==1:
        bus_label[73:89]=[0]*16
        bus_label[88]=1
    if rap_label[76]==1:
        bus_label[73:89]=[0]*16
        bus_label[79]=1
    if rap_label[77]==1:
        bus_label[73:89]=[0]*16
        bus_label[78]=1
    if rap_label[79]==1:
        bus_label[73:89]=[0]*16
        bus_label[83]=1
    if rap_label[80]==1:
        bus_label[73:89]=[0]*16
        bus_label[86]=1
    if rap_label[81]==1:
        bus_label[73:89]=[0]*16
        bus_label[87]=1
    if rap_label[82]==1:
        bus_label[73:89]=[0]*16
        bus_label[77]=1
        
    if rap_label[9]==1:
        bus_label[89:96]=[0]*7
        bus_label[97]=1
    if rap_label[10]==1:
        bus_label[89:96]=[0]*7
        bus_label[100]=1
    
    if rap_label[35]==1:
        bus_label[118:130]=[0]*12
        bus_label[120]=1
    if rap_label[36]==1:
        bus_label[118:130]=[0]*12
        bus_label[119]=1
    if rap_label[37]==1:
        bus_label[118:130]=[0]*12
        bus_label[123]=1
    if rap_label[38]==1:
        bus_label[118:130]=[0]*12
        bus_label[127]=1
    if rap_label[39]==1:
        bus_label[118:130]=[0]*12
        bus_label[121]=1
    if rap_label[40]==1:
        bus_label[118:130]=[0]*12
        bus_label[121]=1
    if rap_label[41]==1:
        bus_label[118:130]=[0]*12
        bus_label[128]=1
        
    if rap_label[0]==0:
        bus_label[108:111]=[0]*3
        bus_label[110]=1
    if rap_label[0]==1:
        bus_label[108:111]=[0]*3
        bus_label[109]=1
    return bus_label
with open('/home/bobby/work/code/bussiness100/prediction_raptestlist.txt') as f:
    bussiness = f.readlines()
    buss_names = [i.split(' ')[0] for i in bussiness]
    buss_labels = [eval(i.split(' ')[1:][0]) for i in bussiness]
    copy_buss_labels = copy.deepcopy(buss_labels)
with open('/home/bobby/work/RAP/minivision_rap/testlist.txt') as f:
    rap = f.readlines()
    rap_labels =[[int(j) for j in i.split(' ')[1:]] for i in rap]
labels_transed = []
for i in range(len(bussiness)):
    temp = classtransfer(buss_labels[i], rap_labels[i])
    labels_transed.append(temp)
    
Flag = 0
if not labels_transed==copy_buss_labels:
    print('modified!')
    Flag = 1
    
if Flag:
    with open('/home/bobby/work/code/bussiness100/prediction_raptestlist_modified.txt', 'w') as f:
        f.writelines([j+' '+str(i)+'\n' for j,i in zip(buss_names,labels_transed)])