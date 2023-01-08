import csv

import cv2
import numpy as np
import os
import pandas as pd
import time


# 读取图像并转换为 HSV 颜色空间
def counting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设定红色和绿色的颜色范围
    lower_red1 = np.array([0, 43, 15])
    upper_red1 = np.array([34, 255, 255])
    lower_red2 = np.array([156, 43, 15])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([78, 43, 15])
    upper_green = np.array([90, 255, 255])

    # 限制图像在红色和绿色范围内
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    # 合并两个 mask
    mask = mask1 + mask2 + mask3

    # 去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # (5,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # 找到苹果的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    countapple = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        k = cv2.isContourConvex(contour)
        if area > 200:
            countapple = countapple + 1
        x, y, w, h = cv2.boundingRect(contour)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return countapple


# 显示结果图像
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# print(countapple)
mainFolder = "val/images"

array_of_img = []


def read_directory(directory_name):
    for filename in os.listdir(r"./" + directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)


read_directory("detect")

data = pd.read_csv("mapping.csv")
# with open("ground_truth.csv", 'r') as f:
#     reader = csv.reader(f)
#     result = list(reader)
# y = result[1]
# y = list(map(int,y))
# print(y)
# y = data['count'].tolist()
# print(y)

returntrue = 0
# print(filename)
# for file in
# start_time = time.time()
# for i in range(0,len(array_of_img)):
#     if counting(array_of_img[i]) ==y[i]:
#         returntrue = returntrue +1
#
# rate = returntrue/len(array_of_img)
# end_time = time.time()
# print(rate)
# print('cost %f second' % (end_time - start_time))
truerate = 0
thres = 10
p = 0
start_time = time.time()

for filename in os.listdir(r"./" + "detect"):

    m = 0
    img = cv2.imread("detect" + "/" + filename)
    k = (((1 - (abs(counting(img) - data[filename]) / data[filename])) < 1).bool() & (
                (1 - (abs(counting(img) - data[filename]) / data[filename])) > 0)).bool()
    if k:
        m = 1 - (abs(counting(img) - data[filename]) / data[filename])
        p = p + 1
        print(m)
        truerate = truerate + m
end_time = time.time()

t = truerate / p

#print('get %f bad' % (p-0))
print('get %f acc' % (t-0))
print('cost %f second' % (end_time - start_time))
