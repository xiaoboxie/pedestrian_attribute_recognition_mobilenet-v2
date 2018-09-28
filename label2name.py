#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:47:19 2018

@author: bobby
"""
import tensorflow as tf
import numpy as np

english = True
def label2name(label):
    if english:
        classname = ["hair color - uncertainty", "hair color - brown", "hair color - orange", "hair color - gray", "hair color - white", "hair color - blue", "hair color - yellow", "hair color - black",
        "Shoe Color - Unsure", "Shoe Color - Stripes", "Shoe Color - Brown / Camel", "Shoe Color - Orange", "Shoe Color - Polka Dot", "Shoe Color - Gray", "Shoe Color - White", "Shoe Color-Pink", "Shoe Color-Purple", "Shoe Color-Red", "Shoe Color-Green", "Shoe Color-Flesh/Nude", "Shoe Color-Flower", "Shoes Color-blue", "shoe color-yellow", "shoe color-black",
        "Head Accessories - Uncertainty", "Head Accessories - Masks", "Head Accessories - Scarves", "Head Accessories - Sunglasses", "Head Accessories - Hats / Headscarves", "Head Accessories - Shawls" , "Head Accessories - None", "Head Accessories - Normal Glasses", "Head Accessories - Colored Glasses",
        "age layer - uncertainty", "age layer - middle age", "age layer - infants", "age layer - juvenile", "age layer - old age", "age layer - youth",
        "Emotion - Uncertainty", "Emotion - General mood", "Emotion - bad mood", "Emotion - good mood",
        "Top sleeves - Unsure", "Top sleeves - Medium and long sleeves", "Top sleeves - Sleeveless / Slings / Tube tops", "Top sleeves - Short sleeves", "Top sleeves - Long sleeves" ,
        "Crowd Attributes - Uncertainty", "Crowd Attributes - Middle Age People", "Crowd Attributes - Pragmatic Salary", "Crowd Attributes - Business Elite", "Crowd Attributes - Infants", "Crowd Attributes - Students", "Crowd Attributes - Home People", "Crowd Attributes - Fashionista", "Crowd Attributes - Seniors",
        "Top Color - Unsure", "Top Color - Stripes", "Top Color - Brown / Camel", "Top Color - Orange", "Top Color - Polka Dot", "Top Color - Grey", "Top Color - White ", "Top Color - Pink", "Top Color - Purple", "Top Color - Red", "Top Color - Green", "Top Color - Flesh / Nude", "Top Color - Color", "Tops Color - Blue", "Top Color - Yellow", "Top Color - Black",
        "Underwear Color - Unsure", "Bottom Color - Stripes", "Bottom Color - Brown / Camel", "Bottom Color - Orange", "Bottom Color - Polka Dot", "Bottom Color - Gray ", "Bottom Color - White", "Under Color - Pink", "Bottom Color - Purple", "Bottom Color - Red", "Bottom Color - Green", "Bottom Color - Flesh / Naked Color", "Bottom color - suit", "Bottom color - blue", "Bottom color - yellow", "Bottom color - black",
        "Tops Style - Uncertainty", "Tops Style - Casual", "Tops Style - Business", "Tops Style - Home", "Tops Style - Workwear", "Tops Style - Simple", "Tops Style - Sports",
        "Hairstyle - Unsure", "Hairstyle - Bald", "Hairstyle - Short Hair", "Hairstyle - Bald", "Hairstyle - Long Hair",
        "Bottom length - Uncertain", "Bottom length - mid-length skirt", "Bottom length - mid-length pants", "Bottom length - short skirt", "Bottom length - shorts", "Bottom length - long skirt", "bottom length - trousers",
        "Gender uncertainty", "gender woman", "gender man",
        "Bottom style - not sure", "Bottom style - casual", "Bottom style - business", "Bottom style - home", "Bottom style - overalls", "Bottom style - simple", " Bottom style - sports",
        "Handbags - Uncertainty", "Luggage Handheld - Shoulder Bag", "Luggage Handheld - Backpack", "Luggage Handheld - Plastic Bag / Shopping Bag", "Luggage Handheld - Stroller / Child", "Handbag Holder -Handbags", "Handbags - diagonal", "Handbags - No", "Handbags - Pockets", "Handbags - Luggage", "Handbags - Shopping Cart", "Handbags - Umbrellas" ]
    else:
        classname = ["发色-不确定", "发色-棕", "发色-橙", "发色-灰", "发色-白", "发色-蓝", "发色-黄", "发色-黑",
        "鞋颜色-不确定", "鞋颜色-条纹", "鞋颜色-棕色/驼色", "鞋颜色-橙色", "鞋颜色-波点", "鞋颜色-灰色", "鞋颜色-白色", "鞋颜色-粉色", "鞋颜色-紫色", "鞋颜色-红色", "鞋颜色-绿色", "鞋颜色-肉色/裸色", "鞋颜色-花色", "鞋颜色-蓝色", "鞋颜色-黄色", "鞋颜色-黑色",
        "头部配饰-不确定", "头部配饰-口罩", "头部配饰-围巾", "头部配饰-太阳镜", "头部配饰-帽子/头巾", "头部配饰-披肩", "头部配饰-无", "头部配饰-普通眼镜", "头部配饰-有色眼镜",
        "年龄层-不确定", "年龄层-中年", "年龄层-婴幼儿", "年龄层-少年", "年龄层-老年", "年龄层-青年",
        "情绪-不确定", "情绪-心情一般", "情绪-心情不好", "情绪-心情好",
        "上装袖长-不确定", "上装袖长-中长袖", "上装袖长-无袖/吊带/抹胸", "上装袖长-短袖", "上装袖长-长袖",
        "人群属性-不确定", "人群属性-中年人", "人群属性-务实工薪", "人群属性-商务精英", "人群属性-婴幼儿", "人群属性-学生", "人群属性-居家人士", "人群属性-时尚达人", "人群属性-老年人",
        "上衣颜色-不确定", "上衣颜色-条纹", "上衣颜色-棕色/驼色", "上衣颜色-橙色", "上衣颜色-波点", "上衣颜色-灰色", "上衣颜色-白色", "上衣颜色-粉色", "上衣颜色-紫色", "上衣颜色-红色", "上衣颜色-绿色", "上衣颜色-肉色/裸色", "上衣颜色-花色", "上衣颜色-蓝色", "上衣颜色-黄色", "上衣颜色-黑色",
        "下衣颜色-不确定", "下衣颜色-条纹", "下衣颜色-棕色/驼色", "下衣颜色-橙色", "下衣颜色-波点", "下衣颜色-灰色", "下衣颜色-白色", "下衣颜色-粉色", "下衣颜色-紫色", "下衣颜色-红色", "下衣颜色-绿色", "下衣颜色-肉色/裸色", "下衣颜色-花色", "下衣颜色-蓝色", "下衣颜色-黄色", "下衣颜色-黑色",
        "上装样式-不确定", "上装样式-休闲", "上装样式-商务", "上装样式-居家", "上装样式-工作服", "上装样式-简约", "上装样式-运动",
        "发型-不确定", "发型-光头", "发型-短发", "发型-秃头", "发型-长发",
        "下装长度-不确定", "下装长度-中长裙", "下装长度-中长裤", "下装长度-短裙", "下装长度-短裤", "下装长度-长裙", "下装长度-长裤",
        "性别不确定", "性别女", "性别男",
        "下装样式-不确定", "下装样式-休闲", "下装样式-商务", "下装样式-居家", "下装样式-工作服", "下装样式-简约", "下装样式-运动",
        "箱包手持-不确定", "箱包手持-单肩背包", "箱包手持-双肩背包", "箱包手持-塑料袋/购物袋", "箱包手持-婴儿车/抱小孩", "箱包手持-手拎箱包", "箱包手持-斜跨包", "箱包手持-无", "箱包手持-腰包", "箱包手持-行李箱", "箱包手持-购物车", "箱包手持-雨伞"]
    name = []
    for i in range(len(label)):
        if label[i]>0.5:
            name.append(classname[i])
    return name
        
def labelstring2label(labelstring):
    return [int(i) for i in list(labelstring.replace(' ',''))]

def getlabelstring(path='/home/bobby/work/dataset/bussiness100/train_new.txt', filename = None):
    with open(path) as f:
        a = f.readlines()
        for i in a:
            if filename in i:
                index = a.index(i)
                return a[index].split(' ')[1:]
            
def score2label(score):
    score = score[0]
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
    a = np.array(a==a.max(), dtype=np.int32)
    b = np.array(b==b.max(), dtype=np.int32)
    c = np.array(c==c.max(), dtype=np.int32)
    d = np.array(d==d.max(), dtype=np.int32)
    e = np.array(e==e.max(), dtype=np.int32)
    f = np.array(f==f.max(), dtype=np.int32)
    if g[-2] > -2.0:
        g = np.zeros(shape=(9,))
        g[-2] = 1               #if fashion score >-2, judge it as fashion
    else:
        g = np.array(g==g.max(), dtype=np.int32)
    h = np.array(h==h.max(), dtype=np.int32)
    i = np.array(i==i.max(), dtype=np.int32)
    j = np.array(j==j.max(), dtype=np.int32)
    k = np.array(k==k.max(), dtype=np.int32)
    l = np.array(l==l.max(), dtype=np.int32)
    m = np.array(m==m.max(), dtype=np.int32)
    n = np.array(n==n.max(), dtype=np.int32)
    o = np.array(o==o.max(), dtype=np.int32)
    pred = np.concatenate((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o))
    return pred
            
            
#seglist = [0]*8
#b = [1]*16
#c = [2]*9
#d = [3]*6
#e = [4]*4
#f = [5]*4
#g = [6]*9
#h = [7]*16
#i = [8]*16
#j = [9]*7
#k = [10]*5
#l = [11]*7
#m = [12]*3
#n = [13]*7
#o = [14]*13
#for p in ['b','c','d','e','f','g','h','i','j','k','l','m','n','o']:
#    seglist.extend(eval(p))           
#segment_ids = tf.Variable(seglist)
#def score2prediction(score):
##    '''
##       0-8 hair color
##       8-24 shoes color
##       24-33 head decoration
##       33-39 age
##       39-43 motion
##       43-47 top sleeve length
##       47-56 pedestrian attribute
##       56-72 top color
##       72-88 bottom color
##       88-95 top style
##       95-100 hairstyle
##       100-107 bottom length
##       107-110 gender
##       110-117 bottom style
##       117-130 handhold
##    '''
#    score = tf.transpose(scorec)
#    score_sigmoid = tf.sigmoid(score)   
#    segment_max_value = tf.segment_max(score_sigmoid,segment_ids)
#    print(tf.shape(segment_max_value))
#    
#    a = score_sigmoid[0:8]
#    b = score_sigmoid[8:24]
#    c = score_sigmoid[24:33]
#    d = score_sigmoid[33:39]
#    e = score_sigmoid[39:43]
#    f = score_sigmoid[43:47]
#    g = score_sigmoid[47:56]
#    h = score_sigmoid[56:72]
#    i = score_sigmoid[72:88]
#    j = score_sigmoid[88:95]
#    k = score_sigmoid[95:100]
#    l = score_sigmoid[100:107]
#    m = score_sigmoid[107:110]
#    n = score_sigmoid[110:117]
#    o = score_sigmoid[117:130]
#    print(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)
#    #a = tf.nn.relu(score[0:8])
##    a_index = tf.argmax(a)
##    b_index = tf.argmax(b)
##    c_index = tf.argmax(c)
##    d_index = tf.argmax(d)
##    e_index = tf.argmax(e)
##    f_index = tf.argmax(f)
##    g_index = tf.argmax(g)
##    h_index = tf.argmax(h)
##    i_index = tf.argmax(i)
##    j_index = tf.argmax(j)
##    k_index = tf.argmax(k)
##    l_index = tf.argmax(l)
##    m_index = tf.argmax(m)
##    n_index = tf.argmax(n)
##    o_index = tf.argmax(o)
#    
#    
#    a = tf.where(tf.equal(a,segment_max_value[0]), a, tf.zeros_like(a))
#    b = tf.where(tf.equal(b,segment_max_value[1]), b, tf.zeros_like(b))
#    c = tf.where(tf.equal(c,segment_max_value[2]), c, tf.zeros_like(c))
#    d = tf.where(tf.equal(d,segment_max_value[3]), d, tf.zeros_like(d))
#    e = tf.where(tf.equal(e,segment_max_value[4]), e, tf.zeros_like(e))
#    f = tf.where(tf.equal(f,segment_max_value[5]), f, tf.zeros_like(f))
#    g = tf.where(tf.equal(g,segment_max_value[6]), g, tf.zeros_like(g))
#    h = tf.where(tf.equal(h,segment_max_value[7]), h, tf.zeros_like(h))
#    i = tf.where(tf.equal(i,segment_max_value[8]), i, tf.zeros_like(i))
#    j = tf.where(tf.equal(j,segment_max_value[9]), j, tf.zeros_like(j))
#    k = tf.where(tf.equal(k,segment_max_value[10]), k, tf.zeros_like(k))
#    l = tf.where(tf.equal(l,segment_max_value[11]), l, tf.zeros_like(l))
#    m = tf.where(tf.equal(m,segment_max_value[12]), m, tf.zeros_like(m))
#    n = tf.where(tf.equal(n,segment_max_value[13]), n, tf.zeros_like(n))
#    o = tf.where(tf.equal(o,segment_max_value[14]), o, tf.zeros_like(o))
#    res = tf.concat((a,b,c,d,e,f,g,h,i,j,k,l,m,n,o), axis=0)
#    return res
#
#import numpy as np
#a = np.random.random(size=(1,130))
#a = tf.Variable(a)
#a = tf.cast(a, tf.float32)
#b = score2prediction(a)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#c = sess.run(b)