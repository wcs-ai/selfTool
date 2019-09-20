#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 
"""
这个文件用来做数据预处理
"""

class dispose(object):
	def __init__(self,data):
		self.data = data

	def del_None(self):
		#delete data of emptey or None
		