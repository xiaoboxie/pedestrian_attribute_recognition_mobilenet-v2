#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:35:21 2018

@author: bobby
"""

import tensorflow as tf
import cv2
from nets.mobilenet import mobilenet_v2
import tensorflow.contrib.slim as slim
import numpy as np
from datagenerator import ImageDataGenerator
from label2name import *
meta_path = 'bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224/model_epoch8.ckpt.meta'
model_path = 'bussiness100_mobilenetv2_1.0_224/checkpoints_mobilenetv2_1.0_224/model_epoch8.ckpt'
img_path = '/home/bobby/work/RAP/RAP_dataset/'
trainlist_path = '/home/bobby/work/RAP/minivision_rap/testlist.txt'
#saver = tf.train.import_meta_graph(meta_path) #import graph
saver = tf.train.Saver()
#variables_to_restore = slim.get_variables_to_restore(exclude=excludeD)
#    # # print(variables_to_restore)
#saver = tf.train.Saver(variables_to_restore)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

num_classes = 130

#img = cv2.imread(img_path)
#img = np.resize(img,new_shape=(1,256,96,3))/255
#img = np.zeros(shape=(1,256,96,3),dtype=np.float32)


with open(trainlist_path) as f:
    trainlist = f.readlines()
    trainlist = [i.split(' ')[0] for i in trainlist]

x = tf.placeholder(tf.float32, shape=(None,256,96,3), name='x')
with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    score, endpoints = mobilenet_v2.mobilenet(x,num_classes,depth_multiplier=1.0,scope='MobilenetV2', reuse=True)
    score = tf.sigmoid(score)
    
#with tf.Graph().as_default():
#    output_graph_def = tf.GraphDef()
#    with open(meta_path, 'rb') as f:
#        output_graph_def.ParseFromString(f.read())
#        _ = tf.iimport_graph_def(output_graph_def, name='')
    
result = []
with tf.Session(config=config) as sess:
    saver.restore(sess, model_path) #import variables
    
#    prob_op = graph.get_operation_by_name('prob')
    #prediction = sess.graph.get_tensor_by_name('MobilenetV2/Logits/output:0')
    #x = sess.graph.get_tensor_by_name('x:0')
    for i in trainlist:
        img = cv2.imread(i)
        img_string = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_string, (96,256))
        img_resized = np.reshape(img_resized,(1,256,96,3))
        img_resized = img_resized / 255
        sco = sess.run(score, feed_dict={x:img_resized})
        pred = score2label(sco)
        label2name(pred)
        result.append(pred)


with open('/home/bobby/work/code/bussiness100/prediction_raptestlist.txt', 'w') as f:
    f.writelines([l+' '+str(i).replace(' ','')+'\n' for l,i in zip(trainlist,[[int(k) for k in j] for j in result])])