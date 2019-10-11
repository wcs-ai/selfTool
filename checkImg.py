#!/usr/bin/python
# # -*- coding: UTF-8 -*-
import cv2
import numpy as np
#检测图片边缘、轮廓、分割...


class check(object):
	"""docstring for check"""
	def __init__(self,img_path):
		#super(check, self).__init__()
		self.img = cv2.imread(img_path,0)
		self.result = ''
	def rough(self):
		ret, thresh = cv2.threshold(self.img, 127, 255, 0)

		image, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for c in image:
			x, y, w, h = cv2.boundingRect(c)
			if w < 30:
				continue
			else:
				cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 5), 2)
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(self.img, [box], 0, (253, 10, 253), 2)


	def show_res(self):
		cv2.imshow('img',self.img)