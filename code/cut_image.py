# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import os
import cv2
import pickle
import mahotas
from itertools import groupby

COLOR_RGB_BLACK = (0, 0, 0)
COLOR_RGB_WHITE = (255, 255, 255)

def _drop_fall(image):
	"""
	对粘连两个字符的图片进行drop fall算法分割
	"""
	#ima = np.array(image)
	#cv2.imshow('image', ima)
	#cv2.waitKey(0)
	
	# 1. 竖直投影统计
	width, height = image.size
	#print "当前待切割图片的 width: %d, height: %d" % (width, height)
	hist_width = [0]*width
	for x in range(width):
		for y in range(height):
			if _is_black(image.getpixel((x, y))):
				hist_width[x] += 1 

	#print "当前的hist_width: %s" % str(hist_width)
		
	# 2. 找到极小值点
	start_x = _get_start_x(hist_width)
	#print( "当前的起始点是: %d" % start_x)

	# 3. 以这个极小值点作为起始滴落点,实施滴水算法
	start_route = []
	for y in range(height):
		start_route.append((0, y))

	end_route = _get_end_route(image, start_x, height, width)
	#print(end_route)
	#filter_end_route = [max(list(k)) for _, k in groupby(end_route, lambda x: x[1])]
	filter_end_route = []
	for _, k in groupby(end_route, lambda x: x[1]):
		tmp = max(list(k))
		filter_end_route.append((tmp[0]+4, tmp[1]))
	# 两个字符的图片，首先得到的是左边那个字符
	img1 = _do_split1(image, start_route, filter_end_route)
	ima = np.array(img1)
	#cv2.imshow('img1', ima)
	#cv2.waitKey(0)

	#img1 = img1.crop((_get_black_border(img1)))


	# 再得到最右边字符
	start_route = map(lambda x: (x[0] + 1, x[1]), filter_end_route)
	end_route = []
	for y in range(height):
		end_route.append((width - 1, y))
	img2 = _do_split1(image, start_route, end_route)
	#ima = np.array(img2)
	#cv2.imshow('img2', ima)
	#cv2.waitKey(0)
	#img2 = img2.crop((_get_black_border1(img2)))

	return [img1, img2]

def _is_black(rgb):
	"""
	: param rgb: tuple (r, g, b) 
	"""
	return True if rgb == COLOR_RGB_BLACK else False

def _get_start_x(hist_width):
	"""
	根据待切割的图片的竖直投影统计hist_width，找到合适的滴水起始点
	hist_width的中间值，前后再取4个值，在这个范围内找最小值
	"""
	mid = int(len(hist_width)/2)
	# 共9个值
	return mid - 4 + np.argmin(hist_width[mid - 4:mid + 5])

def _get_end_route(image, start_x, height, width):
	"""
	获得滴水的路径
	: param start_x: 滴水的起始x位置
	"""
	left_limit = 0
	right_limit = image.size[0] - 1

	end_route = []
	#print "当前的start_x: %d" % start_x
	cur_p = (start_x, 0)
	last_p = cur_p
	end_route.append(cur_p)

	while cur_p[1] < (height - 1):# and cur_p[0] < (width - 1):
		#print('width:', width, cur_p[0])
		sum_n = 0
		maxW = 0 # max Z_j*W_j
		nextX = cur_p[0]
		nextY = cur_p[1]
		for i in range(1, 6):
			#print(cur_p[0], cur_p[1], image.size)
			curW = _get_nearby_pixel_val(image, int(cur_p[0]), int(cur_p[1]), i) * (6 - i)
			sum_n += curW
			if maxW < curW:
				maxW = curW
			
		# 如果全黑，需要看惯性
		if sum_n == 15:
			maxW = 6

		# 如果全白，则默认垂直下落
		if sum_n == 0:
			maxW = 4

		if maxW == 1:
			nextX = cur_p[0] - 1
			nextY = cur_p[1]
		elif maxW == 2:
			nextX = cur_p[0] + 1
			nextY = cur_p[1]
		elif maxW == 4:
			nextX = cur_p[0]
			nextY = cur_p[1] + 1
		elif maxW == 5:
			nextX = cur_p[0] - 1
			nextY = cur_p[1] + 1
		elif maxW == 3:
			nextX = cur_p[0] + 1
			nextY = cur_p[1] + 1
		elif maxW == 6:

			if nextX > cur_p[0]: # 具有向右的惯性
				nextX = cur_p[0] + 1
				nextY = cur_p[1] + 1

			if nextX < cur_p[0]:
				nextX = cur_p[0]
				nextY = cur_p[1] + 1

			if sum_n == 15:
				nextX = cur_p[0] 
				nextY = cur_p[1] + 1
	
		else:
			raise Exception("get a wrong maxW, pls check")

		# 如果出现重复运动
		if last_p[0] == nextX and last_p[1] == nextY:
			if nextX < cur_p[0]:
				maxW = 5
				nextX = cur_p[0] - 1
				nextY = cur_p[1] + 1
			else:
				maxW = 3
				nextX = cur_p[0] + 1
				nextY = cur_p[1] + 1

		last_p = cur_p

		if nextX > right_limit:
			nextX = right_limit
			nextY = cur_p[1] + 1

		if nextX < left_limit:
			nextX = left_limit
			nextY = cur_p[1] + 1

		cur_p = (nextX, nextY)
		end_route.append(cur_p)

	# 返回分割路径
	return end_route

def _get_nearby_pixel_val(image, cx, cy, j):
	if j == 1:
		return 1 if _is_black(image.getpixel((cx - 1, cy + 1))) else 0
	elif j == 2:
		return 1 if _is_black(image.getpixel((cx, cy + 1))) else 0
	elif j == 3:
		if cx+1 >= image.size[0]:
			return 0
		else:
			return 1 if _is_black(image.getpixel((cx + 1 , cy + 1))) else 0
	elif j == 4:
		if cx+1 >= image.size[0]:
			return 0
		else:
			return 1 if _is_black(image.getpixel((cx + 1, cy))) else 0
	elif j == 5:
		return 1 if _is_black(image.getpixel((cx - 1, cy))) else 0
	else:
		raise Exception("what you request is out of nearby range")

def _do_split(source_image, starts, filter_ends):
	"""
	具体实行切割 
	: param starts: 每一行的起始点 tuple of list
	: param ends: 每一行的终止点
	"""
	left = starts[0][0]
	top = starts[0][1]
	right = filter_ends[0][0]
	bottom = filter_ends[0][1]
	#for i in range(len(starts)):
	for i in range(len(starts)):
		left = min(starts[i][0], left)
		top = min(starts[i][1], top)
		right = max(filter_ends[i][0], right)
		bottom = max(filter_ends[i][1], bottom)

	width = right - left + 1
	height = bottom - top + 1

	image = Image.new('RGB', (width, height), COLOR_RGB_WHITE)

	for i in range(height):
		start = starts[i]
		end = filter_ends[i]
		for x in range(start[0], end[0]+1):
			if _is_black(source_image.getpixel((x, start[1]))):
				image.putpixel((x - left, start[1] - top), COLOR_RGB_BLACK)

	return image

def _do_split1(source_image, starts, filter_ends):
	height = source_image.size[1]
	left = right = 0
	#print('111:',starts)
	starts = list(starts)
	#print('222:',starts)
	for i in range(len(starts)):
		right += filter_ends[i][0]
		left += starts[i][0]
	left = int(left/len(starts))
	right = int(right/len(starts))
	#print('left, right',left, right)

	image = Image.new('RGB', (right-left+1, height), COLOR_RGB_WHITE)
	for i in range(height):
		for x in range(left, right+1):
			if _is_black(source_image.getpixel((x, i))):
				#print((x - left, i))
				image.putpixel((x - left, i), COLOR_RGB_BLACK)

	return image


def _get_black_border(image):
	"""
	获取指定图像的内容边界坐标
	:param image: 图像 Image Object
	:return: 图像内容边界坐标tuple (left, top, right, bottom)
	"""
	width, height = image.size
	max_x = width - 1
	max_y = 0
	min_x = width - 1
	min_y = height - 1

	hist = np.zeros(width, dtype=float)
	for x in range(width):
		for y in range(height):
			if image.getpixel((x, y)) == COLOR_RGB_BLACK:
				min_x = min(min_x, x)
				min_y = min(min_y, y)
				max_y = max(max_y, y)
				hist[x] += 1
        
	min_hist = int(min(hist)+np.mean(hist))/2
        #print('min_hist:', min_hist)
	min_list = []
	for i,h in enumerate(hist):
		if h < min_hist and i > width*0.2:
			max_x = i
			break
	#print('max_x:', max_x)
	return min_x, min_y, max_x, max_y + 1

def _get_black_border1(image):
	"""
	获取指定图像的内容边界坐标
	:param image: 图像 Image Object
	:return: 图像内容边界坐标tuple (left, top, right, bottom)
	"""
	width, height = image.size
	max_x = 0
	max_y = 0
	min_x = 0
	min_y = height - 1
	'''
	for y in range(height):
		for x in range(width):
			if image.getpixel((x, y)) == COLOR_RGB_BLACK:
				min_x = min(min_x, x)
				max_x = max(max_x, x) 
				min_y = min(min_y, y)
				max_y = max(max_y, y)
	'''
	hist = np.zeros(width, dtype=float)
	for x in range(width):
		for y in range(height):
			if image.getpixel((x, y)) == COLOR_RGB_BLACK:
				max_x = max(max_x, x)
				min_y = min(min_y, y)
				max_y = max(max_y, y)
				hist[x] += 1
	min_hist = int(min(hist)+np.mean(hist))/2
	for i,h in enumerate(hist):
		if h > min_hist:
			min_x = i
			break
	#print('max_x:', max_x)
	return min_x, min_y, max_x, max_y + 1

def recu(cutim, cutim_list):
	#print('cut:', cutim.size)
	if cutim.size[0] > 45:
 		split_images = _drop_fall(cutim)
	#recu(split_images[0], cutim_list)
	#recu(split_images[1], cutim_list)
	#elif  60 > cutim.size[0] > 25: 
	for split_image in split_images:
		#print('size:',split_image.size)
		#cv2.imshow('split',np.asarray(split_image))
		#cv2.waitKey(0)
		if 65 >= split_image.size[0] > 25: 
			img = np.asarray(split_image)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			cutim_list.append(gray)
			#cv2.imshow('cutttt',img)
			#cv2.waitKey(0)
	return cutim_list


#if __name__ == '__main__':
def get_splitimages(gray):
	#gray = cv2.imread('subim.jpg')
	T = mahotas.thresholding.otsu(gray)
	gray[gray > T] = 255
	gray[gray <= T] = 0
	im = Image.fromarray(gray.astype('uint8')).convert('RGB')
	cutimages = []
	result = recu(im, cutimages)
	#print('cut image: ',len(result))
	return result
