#coding:utf-8
import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
 
def houghtrans(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    rotate_angle = 0
    print(lines[0])
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if x1 == x2 or y1 == y2:
            continue
        t = float(y2 - y1) / (x2 - x1)
        cv2.line(img_copy,(x1,y1),(x2,y2),(255,0,0),1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        print(rotate_angle)
    cv2.imwrite('line.jpg', img_copy)
    print("rotate_angle : "+str(rotate_angle))
    rotate_img = ndimage.rotate(img, rotate_angle, cval=255)
    return rotate_img

'''
if __name__ == '__main__':
    im = cv2.imread('test_images/1.jpg')
    img = houghtrans(im)
'''
