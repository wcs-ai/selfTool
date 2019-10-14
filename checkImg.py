#!/usr/bin/python
# # -*- coding: UTF-8 -*-
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#检测图片边缘、轮廓、分割...


class check(object):
	"""docstring for check"""
	def __init__(self,img):
		#super(check, self).__init__()
		self.img = img
		self.result = ''
		self.ims = ''
	#轮廓检测	
	def rough(self,limit):
		ret, thresh = cv2.threshold(self.img, 127, 255, 0)

		image, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for c in image:
			x, y, w, h = cv2.boundingRect(c)
			if w < limit:
				continue
			else:
				cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 5), 2)
				self.ims = self.img[x:x+w,y:y+h]
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect) 
				box = np.int0(box)
				cv2.drawContours(self.img, [box], 0, (253, 10, 253), 2)

	def grabCut(self):
		mask = np.zeros(self.img.shape[:2],np.uint8)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)
		#限定分割图像的范围
		rect = (10,10,500,580) 
		cv2.grabCut(self.img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		self.img = self.img*mask2[:,:,np.newaxis] 

	def watershed(self):
		gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		kernel = np.ones((3,3),np.uint8)
		#变换，腐蚀去除噪声数据
		opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

		#膨胀,得到的大部分是前景区域
		sure_bg = cv2.dilate(opening,kernel,iterations=3)
		dis_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
		ret,sure_fg = cv2.threshold(dis_transform,0.7*dis_transform.max(),255,0)
		sure_fg = np.uint8(sure_fg)
		#前景与背景有交叉部分，通过相减处理
		unknown = cv2.subtract(sure_bg,sure_fg)
		ret,markers = cv2.connectedComponents(sure_fg)
		markers = markers+1
		markers[unknown==255] = 0
		
		markers = cv2.watershed(self.img,markers)

		self.img[markers==-1] = [255,0,0]



	def show(self):
		cv2.imshow('img',self.img)
		cv2.waitKey()

