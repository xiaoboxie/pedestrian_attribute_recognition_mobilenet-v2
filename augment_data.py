#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:56:24 2018

@author: bobby
"""
        
import cv2
import numpy as np

save_path = '/home/bobby/work/dataset/bussiness100/train_augmentation/'
#save_path = '/home/bobby/work/'
#定义添加椒盐噪声的函数
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
def addGaussianNoise(image,percetage): 
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1]) 
    for i in range(G_NoiseNum): 
        temp_x = np.random.randint(0,h) 
        temp_y = np.random.randint(0,w) 
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] =  np.random.randn(1)[0]
    return G_Noiseimg
#dimming
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy
def rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image
def img_augmentation(path, name_int):        
    img = cv2.imread(path)
    if img.shape[0]>5 and img.shape[1]>5:#avoid bad images
        img_flip = cv2.flip(img,1)#flip
        img_rotation = rotate(img)#rotation
        
        img_noise1 = SaltAndPepper(img, 0.3)
        img_noise2 = addGaussianNoise(img, 0.3)
        
        img_brighter = brighter(img)
        img_darker = darker(img)
    
        cv2.imwrite(save_path+'%s' %str(name_int)+'.jpg', img_flip)
        cv2.imwrite(save_path+'%s' %str(name_int+1)+'.jpg', img_rotation)
        cv2.imwrite(save_path+'%s' %str(name_int+2)+'.jpg', img_noise1)
        cv2.imwrite(save_path+'%s' %str(name_int+3)+'.jpg', img_noise2)
        cv2.imwrite(save_path+'%s' %str(name_int+4)+'.jpg', img_brighter)
        cv2.imwrite(save_path+'%s' %str(name_int+5)+'.jpg', img_darker)

with open('/home/bobby/work/dataset/bussiness100/rare_class.txt') as f:
    rare_attribute = f.readlines()
    rare_attribute_list = [i.split(',')[0].strip("'") for i in rare_attribute]

with open('/home/bobby/work/dataset/bussiness100/classname') as f:
    temp = f.read()
    attribute_list = eval(temp.split(']')[0]+']')
    
rare_attribute_index = [attribute_list.index(i) for i in rare_attribute_list]

with open('/home/bobby/work/dataset/bussiness100/train.txt') as f:
    dataset = f.readlines()
    labels_list = [[eval(j.strip()) for j in i.split(' ')[1:]] for i in dataset]
    name_list = [i.split(' ')[0].replace('./','/home/bobby/work/dataset/bussiness100/') for i in dataset]

add_list = []
name_int = 200000
print('Start Data Augmentation...')
print('every 6 imgs are shown one time')
for each_index in rare_attribute_index:
    append_list = []
    for label in labels_list:
        if label[each_index] == 1:
            index = labels_list.index(label)
            img_augmentation(name_list[index],name_int=name_int)

            label_str = ''
            for i in label:
                label_str += (' '+str(i))
            for i in range(6):
                label_for_write = save_path+'%s'%(name_int+i)+'.jpg'+label_str
                
                add_list.append(label_for_write)
            
            print(label_for_write)
            name_int += 6
with open('/home/bobby/work/dataset/bussiness100/train_augmentation.txt', 'w') as f:
    f.writelines([i+'\n' for i in add_list])
#img_augmentation('/home/bobby/work/0.jpg',0)
#def _parse_function_train(self, filename, label):
#    """Input parser for samples of the training set."""
#    # convert label number into one-hot-encoding
#    # one_hot = tf.one_hot(label, self.num_classes)
#    # print("label {}".format(one_hot))
#    # IMAGE_WIDTH = 1280
#    # IMAGE_HEIGHT = 720
#    # for business 100 dataset
#    IMAGE_WIDTH = 3024
#    IMAGE_HEIGHT = 4032
#
#    label = tf.cast(label, tf.float32)
#    one_hot = label
#    # load and preprocess the image
#    img_string = tf.read_file(filename)
#    img_decoded = tf.image.decode_png(img_string, channels=3)
#
#    """
#    Dataaugmentation comes here.
#    """
#    img_resized = tf.image.resize_images(img_decoded, [INPUT_HEIGHT + 12, INPUT_WIDTH + 10])
#    img_resized_croped = tf.random_crop(img_resized, [INPUT_HEIGHT, INPUT_WIDTH, 3])
#    img_decoded = tf.image.random_flip_left_right(img_resized_croped, seed=30)
#
#    img_decoded = tf.image.random_brightness(img_decoded, max_delta=32. / 255.)
#    ## 开始下面两行代码之后训练速度慢很多,大约会慢6倍,具体那一行还不确定.
#    # img_decoded = tf.image.random_saturation(img_decoded, lower=0.5, upper=1.5)
#    # img_decoded = tf.image.random_hue(img_decoded, max_delta=0.2)
#
#    # img_resized = tf.image.resize_images(img_decoded, [INPUT_HEIGHT, INPUT_WIDTH])
#    img_float32 = tf.cast(img_decoded, tf.float32)
#    img_rgb = tf.div(img_float32, 255.0)
#
#    # img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
#
#    return img_rgb, one_hot