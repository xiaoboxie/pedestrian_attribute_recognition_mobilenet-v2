#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:47:24 2018

@author: bobby
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:07:23 2018

@author: bobby
"""
from PIL import Image
import re
import numpy as np
from keras.layers import Dense,Conv2D,MaxPooling2D,Input,BatchNormalization,Activation,Flatten
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TensorBoard,ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import load_model
import os
import math
trainlist_path = '/home/bobby/work/dataset/bussiness100/train.txt'
testlist_path = '/home/bobby/work/dataset/bussiness100/val.txt'
trainset_path = '/home/bobby/work/dataset/train/'
testset_path = '/home/bobby/work/dataset/test/'
if not os.path.exists('/home/bobby/work/code/bussiness100/6_22_checkpoint/'):
    os.mkdir('/home/bobby/work/code/business100/6_22_checkpoint/')
checkpoint_path = '/home/bobby/work/code/bussiness100/6_22_checkpoint/'

def read_imgs(path):
    img = Image.open(path)
    img = img.resize((227,227))
    img = np.asarray(img)
    return img

def split_set_label(setlist):
    set1=[]
    set2=[]
    path_compiler = re.compile('.+[jpg,png]')
    label_compiler = re.compile('(?<=[jpg,png] ).+')
    for i in setlist:
        imgpath = re.findall(path_compiler, i)[0]
        imgpath = imgpath.replace('./', '/home/bobby/work/dataset/bussiness100/')
        img = read_imgs(imgpath)
        set1.append(img)
        label_string = re.findall(label_compiler, i)[0].replace(' ','')
        label = [eval(i) for i in label_string]
        set2.append(label)
    return set1,set2

def img_normalization(img):
    img = img.astype('float64')/255
    img_mean = np.mean(img, axis=0)
    img -= img_mean
    return img

#read data list
with open(trainlist_path) as f:
    trainlist_all = f.readlines()
with open(testlist_path) as f:
    testlist = f.readlines()

cnt = 0
ratio_dic = {}
batchsize = 64
steps_per_epoch = math.ceil(len(trainlist_all)/batchsize)
def load_trainset(batchsize=batchsize):
    global cnt
    global ratio_dic
    #load images
    while True:
        for _ in range(steps_per_epoch):
            trainlist = trainlist_all[cnt:cnt+batchsize]
            cnt += batchsize
            if trainlist == []:
                cnt = 0
                break
            train_set, train_labels = split_set_label(trainlist)
            
            train_set = np.array(train_set)
            train_labels = np.array(train_labels)

            ratio_dic = {i:np.exp(-np.sum(train_labels.T[:]==1)/(train_labels.shape[0])) for i in range(130)}
            train_set = img_normalization(train_set)

            yield (train_set,train_labels)

def load_testset():
    test_set, test_labels = split_set_label(testlist)
    test_set = np.array(test_set)
    test_labels = np.array(test_labels)
    test_set = img_normalization(test_set)
    return test_set,test_labels
test_set, test_labels = load_testset()
#normalization


#parameters
initial_lr = 0.001
weight_decay = 0.005

#use gpu
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
#construct network
inputs = Input(shape=(227,227,3))

model = Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='valid')(inputs) 
model = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model =Conv2D(filters=256, kernel_size=(5,5), strides=2, padding='same')(model)
model = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Conv2D(filters=384, kernel_size=(3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Conv2D(filters=384, kernel_size=(3,3), padding='same')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Conv2D(filters=256,kernel_size=(3,3), padding='same')(model)
model = MaxPooling2D(strides=2, padding='valid')(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Flatten()(model)
model = Dense(4096, activation='relu')(model)
model = Dense(4096, activation='relu')(model)
outputs = Dense(130, activation='sigmoid')(model)

model = Model(inputs=inputs, outputs=outputs)

sgd = optimizers.SGD(lr=initial_lr, momentum=0.9, decay=weight_decay, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc_exam'])
model.summary()

checkpoint_name = checkpoint_path+'130bestmodel.h5'
#define callbacks
def schedule(epoch):
    lr = 1e-3
    if 0<epoch<=50:
        lr=1e-1
    elif 50<epoch<=100:
        lr=1e-2
    elif 100<epoch<=150:
        lr=1e-3
    else:
        lr=0.5e-3
    print('The learning is %s now' %lr)
    return lr
checkpoint = ModelCheckpoint(filepath=checkpoint_name, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
#lr_scheduler = LearningRateScheduler(schedule=schedule, verbose=1)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, mode='min', cooldown=0, min_lr=0.5e-6)
callbacks=[checkpoint,earlystopping,reducelr]

is_load_model = True
if is_load_model:
    if os.path.exists(checkpoint_name):
        model = load_model(checkpoint_name)
model.fit_generator(load_trainset(),
          validation_data=(test_set,test_labels),
          steps_per_epoch = steps_per_epoch,
          class_weight=ratio_dic,
          epochs=200, 
          verbose=1,
          shuffle=True,
          callbacks=callbacks)
        
def metrics_all_attribute(y_test=test_labels):
    pred = model.predict(x=test_set)
    pred = pred.round()
    pred = pred.astype('int')
    y_test = y_test.astype('int')
    TP = np.bitwise_and(y_test, pred).sum()
    FN_FP = np.bitwise_xor(y_test, pred).sum()
    TP_FP = pred.sum()
    TP_FN = y_test.sum()
    accuracy = TP/FN_FP
    precision = TP/TP_FP
    recall = TP/TP_FN
    return accuracy,precision,recall

