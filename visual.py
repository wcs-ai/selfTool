#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
这个文件用来绘图
"""
import numpy as np
import matplotlib.pyplot as plt

class Draw(object):
	def __init__(self,figure=[30,30],title='Draw',color='blue',subplot=[1,1,1]):
		self._params = {
			"title":title,
			"figure":figure,
			"color":'blue',
			"subplot":subplot
		}
		plt.title(title)
		plt.figure(figsize=figure)

	def line(self,x,y):
		can_iterable = [list,tuple,np.ndarray]
		if type(y[0]) in can_iterable:
			for i,con in enumerate(y):
				s = len(y)
				plt.subplot(s,1,i+1)
				plt.plot(x,con,color=self._params['color'])
		else:
			plt.plot(x,y,color=self._params['color'])
		plt.show()


def bar(x,y):
    plt.title('data bar')
    plt.figure(figsize=(50,80))
    plt.bar(x,y)
    plt.show()

