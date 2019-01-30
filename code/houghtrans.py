#coding:utf-8
import os
import cv2
import math
import random
from math import *
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

def rotateImage(src, degree):
    # 旋转中心为图像中心
    height, width = src.shape[:2]
    h_new=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    w_new=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
    #print('ratate: ', h_new, w_new)
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((width/2.0, height/2.0), degree, 1)

    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w_new, h_new), borderValue=(255, 255, 255))
    return rotate


def DegreeTrans(theta):
    res = theta / np.pi * 180
    return res

def houghtrans(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 200, apertureSize=3)
    #cv2.imshow('edges', edges)
    #cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 0, 1000, 10)
 
    print(lines.shape)
    sum_theta = 0
    count = 0
    rotate_angle = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            if x1 == x2 or y1 == y2:
                continue
            t = float(y2 - y1) / (x2 - x1)
            if t > 1 or t < -1:
                continue
            #print(t, math.degrees(math.atan(t)))
            sum_theta += t
            count += 1
            #cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if count == 0:
        rotate_angle = 0
    else:
        rotate_angle = math.degrees(math.atan(sum_theta / count))
    #cv2.imwrite('line_1.jpg',img_copy)
    #print("rotate_angle : "+str(rotate_angle))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle
    print("rotate_angle : "+str(rotate_angle))
    if abs(rotate_angle) > 2:
        if rotate_angle < 0:
            rotate_angle = rotate_angle/2.0
        else:
            rotate_angle = rotate_angle/2.0
    print("rotate_angle : "+str(rotate_angle))
    #rotate_img = ndimage.rotate(img, rotate_angle, cval=255)
    rotate_img = rotateImage(img, rotate_angle)
    #cv2.imwrite('ratate_1.jpg', rotate_img)
    return rotate_img





def houghtrans_old(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    
    print(lines.shape)
    rotate_angle = 0
    print('lines[0]: ',lines[0])
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        if x1 == x2 or y1 == y2:
            continue   
        theta = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(theta))
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        print("rotate_angle : "+str(rotate_angle))

    cv2.imwrite('line.jpg', img_copy)
    rotate_img = ndimage.rotate(img, rotate_angle, cval=255)
    #rotate_img = rotateImage(img, rotate_angle)
    cv2.imwrite('ratate.jpg', rotate_img)
    return rotate_img

if __name__ == '__main__':
    im = cv2.imread('12.jpg')
    img = houghtrans_old(im)
    img_1 = houghtrans(im)

