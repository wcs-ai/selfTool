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
		rect = (100,50,421,378)
		img = cv2.grabCut(self.img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		
		#self.img = img*mask2[:,:,np.newaxis]
		ig = [list(img[0]),list(img[1]),list(img[2])]
		
		cv2.imshow('img',ig)
		cv2.waitKey()
		# plt.subplot(121)
		# plt.imshow(img)
		# plt.title('grab')
		# plt.subplot(122)
		# plt.imshow(self.img)
		# plt.title('org')
		# plt.show()


	def show_res(self):
		cv2.imshow('img',self.img)
		cv2.waitKey()

