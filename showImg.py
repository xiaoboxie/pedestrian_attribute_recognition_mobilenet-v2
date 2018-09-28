#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:26:05 2018

@author: bobby
"""
import random


from PIL import Image,ImageDraw,ImageFont
import time
from pykeyboard import PyKeyboard

with open('/home/bobby/work/dataset/bussiness100/val_new.txt') as f:
    file = f.readlines()
with open('/home/bobby/work/dataset/bussiness100/classname') as f:
    classname = f.read()
    classname = eval(classname.split('\n\n')[0])


while True:
    labeln = random.choice(range(0,130)) #choose which label to show
    labeln = 55
    labelname = classname[labeln]
    i = random.choice(file)
    if i.split()[1:][labeln]=='1':
        fp = open(i.replace('./','/home/bobby/work/dataset/bussiness100/').split()[0],'rb')
        a = Image.open(fp)
        ttfont = ImageFont.truetype(font='pixel.ttf',size=20)
        draw = ImageDraw.Draw(a)
        draw.text((0.1*a.size[0],0.1*a.size[1]),labelname,fill=(0,255,0), font=ttfont)
        
        a.show()
        time.sleep(2)
        k = PyKeyboard()
        k.tap_key(k.escape_key)
        fp.close()