#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 
"""
这个文件用来做数据预处理
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing


class dispose(object):
	"""docstring for ClassName"""
	def __init__(self, data):
		#super(ClassName, self).__init__()
		self.data = np.array(data)
		self.take_data_after = ''
		sp = np.shape(data)
		assert len(sp)==2,'sape must be 2'
		self.shape = sp


	#unusual number to None
	def figure_to_None(self,columns='all',algorithm='gaussan'):
		clm = range(len(self.shape[1])) if columns=='all' else columns
		#大于3倍平方差的变为缺失值
		if algorithm=='gaussan':
			for i in clm:
				std = np.std(self.data[:,i])
				self.data[self.data[:,i]>3*std] = None


	#take missing value			
	def takeNone(self,method='del'):
		arr_drop = None
		arr = pd.DataFrame(self.data)
		#delete the row that include None
		if method=='del':
			for j in range(len(self.shape[1])):
				arr_drop = arr.dropna(axis=0)
		elif method=='lagrange':
			#notnull()将数据转为True，False表示
			arr_null = arr.notnull()
			for n in range(len(self.shape[1])):
				print(n)
		self.take_data_after = arr_drop




#数据规范化
def norm_data(data,query_shape,method='norm'):
	shape = np.shape(data)
	take_data = []
	dt = np.reshape(data,query_shape)

	if len(shape)>2:
		for d in dt:
			take_data.append(preprocessing.normalize(d,norm='l2'))
	else:
		take_data = preprocessing.normalize(dt,norm='l2')
	res_data = np.reshape(take_data,shape)
	return res_data