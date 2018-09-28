#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:23:11 2018

@author: bobby
"""
import random
import numpy as np
import cv2
with open('/home/bobby/work/dataset/bussiness100/train_new_copy.txt') as f:
    dataset = f.readlines()
    labelset = [[int(j) for j in i.split()[1:]] for i in dataset]
#    a = score[0:8]
#    b = score[8:24]
#    c = score[24:33]
#    d = score[33:39]
#    e = score[39:43]
#    f = score[43:48]
#    g = score[48:57]
#    h = score[57:73]
#    i = score[73:89]
#    j = score[89:96]
#    k = score[96:101]
#    l = score[101:108]
#    m = score[108:111]
#    n = score[111:118]
#    o = score[118:130]
count = 0
count1 = 0
newdataset = []

def SaltAndPepper(src,percetage):  
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1]) 
    for i in range(SP_NoiseNum): 
        randR=np.random.randint(0,src.shape[0]-1) 
        randG=np.random.randint(0,src.shape[1]-1) 
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0: 
            SP_NoiseImg[randR,randG,randB]=0 
        else: 
            SP_NoiseImg[randR,randG,randB]=255 
    return SP_NoiseImg 
#定义添加高斯噪声的函数 
def GaussianNoise(image,percetage): 
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] =  np.random.randn(1)[0]
    return G_Noiseimg

rootdir = '/home/bobby/work/dataset/bussiness100/'

for i in range(len(labelset)):
    label = labelset[i]
    if label[7]==1 and (label[8]==1 or label[14]==1 or label[23]==1)\
        and (label[24]==1 or label[30]==1 or label[31]==1)\
        and (label[33]==1 or label[34]==1 or label[38]==1)\
        and (label[39]==1 or label[40]==1)\
        and (label[47]==1)\
        and (label[48]==1 or label[50]==1)\
        and (label[62]==1 or label[70]==1 or label[72]==1)\
        and (label[86]==1 or label[88]==1)\
        and (label[90]==1)\
        and (label[98]==1 or label[100]==1)\
        and (label[107]==1)\
        and (label[109]==1 or label[110]==1)\
        and (label[112]==1 or label[116]==1)\
        and (label[118]==1 or label[124]==1 or label[125]==1, label[121]==1):
        if random.choice([1,0]):
            
            newdataset.append(dataset[i])
        count += 1
    else:
        newdataset.append(dataset[i])
        if (label[2]==1 or label[3]==1 or label[4]==1 or label[5]==1)\
            or (label[27]==1 or label[29]==1 or label[32]==1)\
            or (label[41]==1)\
            or (label[45]==1)\
            or (label[57]==1 or label[61]==1)\
            or (label[92]==1 or label[95]==1)\
            or (label[97]==1 or label[99]==1)\
            or (label[105]==1 or label[106]==1)\
            or (label[113]==1 or label[114]==1 or label[115]==1)\
            or (label[126]==1 or label[127]==1 or label[128]==1 or label[129]==1)\
            and (label[9]==1 or label[11]==1 or label[12]==1 or label[16]==1 or\
                 label[18]==1 or label[19]==1 or label[20]==1 or label[22]==1)\
            and (label[74]==1 or label[76]==1 or label[77]==1 or label[79]==1\
                 or label[80]==1 or label[81]==1 or label[82]==1 or label[83]==1\
                 or label[84]==1 or label[85]==1 or label[87]==1):
            
            
#            img = cv2.imread(dataset[i].split()[0].replace('./',rootdir))
#            salt = SaltAndPepper(img, 0.2)
#            gauss = GaussianNoise(img, 0.2)
#            sname =rootdir+'AugmentationMinor/'+str(i)+'_s'+'.jpg'
#            gname = rootdir+'AugmentationMinor/'+str(i)+'_g'+'.jpg'
#            cv2.imwrite(sname, salt)
#            cv2.imwrite(gname, gauss)
#            saltlabel = sname+' '+' '.join(dataset[i].split()[1:])
#            gausslabel = gname+' '+' '.join(dataset[i].split()[1:])
#            newdataset.append(saltlabel)
#            newdataset.append(gausslabel)
#            newdataset.append(dataset[i])

            count1 += 1
print('subsampling  ',count)
print('upsampling  ',count1)
print(len(newdataset))
random.shuffle(newdataset)
with open('/home/bobby/work/dataset/bussiness100/train_undersample.txt','w') as f:
    f.writelines([i.strip()+'\n' for i in newdataset])
    