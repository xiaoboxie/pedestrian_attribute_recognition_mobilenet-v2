#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:47:32 2018

@author: bobby
"""

import tensorflow as tf
import cv2
from nets.mobilenet import mobilenet_v2
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image
from label2name import *
import random
import time
from pykeyboard import PyKeyboard
from PIL import ImageFont,ImageDraw
import getch
import easygui
meta_path = 'bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224/model_epoch9.ckpt.meta'
model_path = 'bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224/model_epoch9.ckpt'
#model_path = 'bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224_best/model_epoch9.ckpt'
datasetpath = '/home/bobby/work/dataset/bussiness100/train_new_copy.txt'
#trainlist_path = '/home/bobby/work/RAP/minivision_rap/trainlist_shuffle.txt'
#saver = tf.train.import_meta_graph(meta_path) #import graph
tf.reset_default_graph()
#variables_to_restore = slim.get_variables_to_restore(exclude=excludeD)
#    # # print(variables_to_restore)
#saver = tf.train.Saver(variables_to_restore)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

num_classes = 130

#img = cv2.imread(img_path)
#img = np.resize(img,new_shape=(1,256,96,3))/255
#img = np.zeros(shape=(1,256,96,3),dtype=np.float32)


x = tf.placeholder(tf.float32, shape=(None,256,96,3), name='x')

with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    score, endpoints = mobilenet_v2.mobilenet(x,num_classes,depth_multiplier=1.0,scope='MobilenetV2', reuse=False)
    scorej = tf.sigmoid(score)
#    prediction = tf.round(prediction)
#    prediction = score2prediction(score)
#with tf.Graph().as_default():
#    output_graph_def = tf.GraphDef()
#    with open(meta_path, 'rb') as f:
#        output_graph_def.ParseFromString(f.read())
#        _ = tf.iimport_graph_def(output_graph_def, name='')
    
result = []
saver = tf.train.Saver()

def parse_function_inference(filename):
    """Input parser for samples of the validation/test set."""
    # convert label number into one-hot-encoding
    # one_hot = tf.one_hot(label, self.num_classes)

    # load and preprocess the image
    img_string = cv2.imread(filename)
#    cv2.imshow('',img_string)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    img_string = cv2.cvtColor(img_string, cv2.COLOR_BGR2RGB)
    
#    cv2.imshow('123',img_string)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    cv2.waitKey(1)
#    print(img_string.shape)
    #img_string = img_string.reshape((img_string.shape[1], img_string.shape[0],3))
    img_resized = cv2.resize(img_string, (96,256))
#    cv2.imshow('123',img_resized)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    cv2.waitKey(1)
    img_resized = np.reshape(img_resized,(1,256,96,3))
    img_resized = img_resized / 255
    """
    Dataaugmentation comes here.
    """
    # img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
    img_centered = img_resized

    # RGB -> BGR
    # img_bgr = img_centered[:, :, ::-1]
    

    return img_centered

def personatt(score):
    attname = ["人群属性-不确定", "人群属性-中年人", "人群属性-务实工薪", \
               "人群属性-商务精英", "人群属性-婴幼儿", "人群属性-学生", \
               "人群属性-居家人士", "人群属性-时尚达人", "人群属性-老年人"]
    logits_personatt = score[48:57]
    assert(len(attname)==len(logits_personatt))
    attdict = {}
    for i in range(len(attname)):
        attdict[attname[i]] = float('%.2f' %(logits_personatt[i]))
    attdict = sorted(attdict.items(), key=lambda item:item[1])
    return attdict
        

with open(datasetpath) as f:
    dataset = f.readlines()

with tf.Session(config=config) as sess:
    saver.restore(sess, model_path) #import variables
    
#    prob_op = graph.get_operation_by_name('prob')
    #prediction = sess.graph.get_tensor_by_name('MobilenetV2/Logits/output:0')
    #x = sess.graph.get_tensor_by_name('x:0')
#    img = cv2.imread(img_path)
    rightcount = 0
    faultcount = 0
    is_random_choose = False
    i = 0
    is_showimg = True
    while i<len(dataset):
        peopleatt = ["人群属性-不确定", "人群属性-中年人", "人群属性-务实工薪", \
           "人群属性-商务精英", "人群属性-婴幼儿", "人群属性-学生", \
           "人群属性-居家人士", "人群属性-时尚达人", "人群属性-老年人"]
        if is_random_choose:
            img_path_label = random.choice([i.replace('./','/home/bobby/work/dataset/bussiness100/') for i in dataset])
        else:
            img_path_label = dataset[i].replace('./','/home/bobby/work/dataset/bussiness100/')
            i += 1
        img_path = img_path_label.split()[0]
        img_path = '/home/bobby/keras-yolo3/3.jpg'
        groudtruth = img_path_label.split()[1:]
        peopleattlabel = groudtruth[48:57]
#        if peopleattlabel[-2] != '1':
#            continue
#        img_path = '/home/bobby/桌面/33.jpg'
        img = parse_function_inference(img_path)
        
        img_path = open(img_path,'rb')
        img_show = Image.open(img_path)
        img_arr = np.array(img_show)
#        print(img_arr.shape)
#        for i in range(3):
#            img_show = np.rot90(img_show)
#        img_show = Image.fromarray(img_show)
        #img_show = img_show.resize((400,1000))
        #img_show = img_show.resize((96,256))
        
    #    img = np.resize(img, new_shape=(1,256,96,3))/255
        score1 = sess.run(score, feed_dict={x:img})
        pred = score2label(score1)
        if score1[0][55]<-2.0:
            continue
        
        if peopleattlabel[-2] == '1':
            print('right')
            rightcount += 1
        else:
            print('fault', end=' ')
            faultcount += 1
        
        #print(pred)
    #    result.append(pred)
        if is_showimg:
            print(peopleatt[peopleattlabel.index('1')])
            print(personatt(score1[0]))
            res = label2name(pred)
            res = '\n'.join(res)
            ttfont = ImageFont.truetype(font='pixel.ttf',size=(img_arr.shape[1]//20))
            draw = ImageDraw.Draw(img_show)
            draw.text((0.1*img_show.size[0],0.1*img_show.size[1]),res,fill=(0,255,0), font=ttfont)
            
            
            img_show.show()
            a = easygui.buttonbox(msg='next image', choices=('Y'))
            if a != 'Y':
                break
    #        time.sleep(5)
            
            
            k = PyKeyboard()
            k.tap_key(k.escape_key)
            
            img_show.close()
    print('-'*30)
    print('right count:', rightcount)
    print('fault count:', faultcount)