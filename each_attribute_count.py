#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:29:48 2018

@author: bobby
"""
index1 = [47, 107, 90, 7, 112, 88, 109, 23, 38, 30, 40, 98, 50, 39, 72, 100, 110, 125, 48, 24, 118, 33, 86, 31, 116, 34, 14, 124, 62]
index2 = [121, 70, 49, 96, 42, 120, 13, 8, 63, 1, 123, 66, 59, 94, 10, 119, 78, 111, 53, 101, 0, 67, 28, 64, 108, 69, 35, 73, 55, 103,\
          52, 43, 17, 54, 71, 104, 36, 26, 37, 21, 56, 6, 58, 89, 68, 122, 44, 102, 93, 65, 75, 95, 57, 117, 46, 15]
index3 = [91, 25, 60, 79, 85, 61, 84, 113, 4, 115, 82, 16, 18, 3, 51, 106, 128, 105, 22, 99, 97, 83, 41, 19, 87, 20, 9, 80, 32, 74,\
          127, 92, 27, 77, 81, 45, 11, 114, 129, 12, 126, 29, 76, 5, 2]
index_test = [index2[2],index2[18],index2[28],index2[30],index2[33],index2[40]]
with open('/home/bobby/work/dataset/bussiness100/test.txt') as f:
    dataset = f.readlines()
    labels_list = [[int(j) for j in i.split(' ')[1:]] for i in dataset]

count_list = []
for i in range(len(labels_list[0])):
    count = sum([j[i] for j in labels_list])
    count_list.append(count)

with open('/home/bobby/work/dataset/bussiness100/classname') as f:
    classname = f.read()
    classname_list = eval(classname.split('\n\n')[0])
    #classname_list = [classname_list[i] for i in index_test]#for split
print(count_list)
with open('/home/bobby/work/code/bussiness100/each_attribute_count.txt', 'w') as f:
    f.writelines([str(i)+' '+j+str(k)+'\n' for i,j,k in zip(count_list, classname_list, range(len(classname_list)))])
    
print(classname_list)
print([i/len(dataset) for i in count_list])

