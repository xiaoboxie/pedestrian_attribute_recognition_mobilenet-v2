import cv2
import os
import numpy as np

valfile = open("train.txt")
valfilenew = open("train_new.txt", "w")

cnt = 0
for i in valfile.readlines():
    line = i
    i = i.split(" ")[0]
    image = cv2.imread(i)
    h,w,c = image.shape
    sum = np.sum(image)
    # print(sum)
    if sum < 100 or w < 50 or h < 50:   # 去掉空白图像和小图像
        cnt = cnt + 1
        print(cnt)
        # cv2.imshow("demo", image)
        # cv2.waitKey(0)
    else:
        valfilenew.write(line)

print(cnt)


