#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:39:35 2018

@author: bobby
"""
xml_path = '/home/bobby/work/dataset/标注数据修改/00000050.xml'
output_path = xml_path

atts =['发色','鞋颜色','头部配饰','年龄层','情绪','上装袖长','上衣颜色',
       '下衣颜色','上装样式','发型','下装长度','性别','下装样式','箱包手持', '人群属性']
from xml.etree import ElementTree as ET

tree = ET.parse(xml_path)

root = tree.getroot()

for eachelement in root.findall('object'):
    for i in range(14):
        element = eachelement.find(atts[i])
        eachelement.remove(element)
    attelement = eachelement.find(atts[-1])
    nameelement = eachelement.find('name')
    nameelement.text = attelement.text
    eachelement.remove(attelement)
ET.dump(root)
tree.write(output_path,"UTF-8")