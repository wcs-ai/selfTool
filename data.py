#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 
"""
这个文件用来做数据预处理
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json

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


#求两点间距离
def point_distance(x,y):
    a = x if type(x)==np.ndarray else np.array(x)
    b = y if type(y)==np.ndarray else np.array(y)

    c = np.sum(np.square(a-b))
    return np.sqrt(c)

#数据规范化
def norm_data(data,query_shape,method='norm'):
	shape = np.shape(data)
	take_data = []
	dt = np.reshape(data,query_shape)

	if method=='norm':
		scaler = preprocessing.normalize
	elif method=='max-min':
		scaler = preprocessing.MinMaxScaler()
	elif method=='qt':
		scaler = preprocessing.QuantileTransformer()

	#可批量归一化数据
	if len(shape)>2:
		for d in dt:
			if method=='norm':
				take_data.append(scaler(d,norm='l2'))
			else:
				take_data.append(scaler.fit_transform(d))
	else:
		if method=='norm':
			take_data = scaler(d,norm='l2')
		else:
			take_data = scaler.fit_transform(d)
	#还原数据形状
	res_data = np.reshape(take_data,shape)
	return res_data


#将腾讯的词向量文件分成9个json文件
def divide_tencent_vector(tencent_path):
    file = open(tencent_path,'r',encoding='utf-8')
    save_path = "data/tencent_vector"
    WORDER_NUM = 1000000
    #1000000
    vector = []
    i = 5
    j = 0
    idx = 0

    tencent = {
        "f1":{},
        "f2":{},
        "f3":{},
        "f4":{},
        "f5":{},
        "f6":{},
        "f7":{},
        "f8":{},
        "f9":{},
        "f10":{}
    }

    for item in file:
        if j>WORDER_NUM*4:
            arr = item.split()
            arr_one = arr.pop(0)
            #部分一行有两个中文词
            one = len(arr) - 200
            for n in range(one):
                arr_one = arr.pop(0)

            vector = list(map(float,arr))
                
            idx = "f" + str(i)
            tencent[idx][arr_one] = vector
        else:
            pass

        if i < 9:
            if j>WORDER_NUM*i:
                save_name = save_path + str(i) + ".json"
                with open(save_name,'w') as jsonObj:
                    json.dump(tencent[idx],jsonObj)

                    i = i + 1
                del tencent[idx]
            else:
                pass       
        else:
            pass
        j = j + 1

    last_name = save_path + str(i) + ".json"
    with open(last_name,'w') as last:
        json.dump(tencent[idx],last)


#利用9个腾讯词向量json文件将汉字转为词向量,不占用内存的使用方法(内存低于16g时使用)
def words_to_vector(open_path,save_path,tencent_path):
    data_in = np.load(open_path,allow_pickle=True)

    #按照原数据结构，构建一个维数完全相同的数组，即使原数据未对齐
    zero_index = []
    for u,val in enumerate(data_in):
        zero_index.append([])
        for t in val:
            zero_index[u].append(0)

    #save data that tencent's vector
    words_obj = {}
    tencent_file = {
        "t1":1,
        "t2":2,
        "t3":3,
        "t4":1,
        "t5":2,
        "t6":3,
        "t7":1,
        "t8":2,
        "t9":3,
    }

    def read_tencent(ord):
        path = tencent_path+'tc'+ str(ord) +'.json'
        with open(path,'r') as f:
            data = json.load(f)
        return data
    #将所有要查找的词都放到words_obj中去
    for a,item in enumerate(data_in):
        for b,word in enumerate(item):
            if word not in words_obj:
                words_obj[word] = []

            words_obj[word].append([a,b])

    #逐个打开腾讯文件
    for f in range(1,10):
        key = 't' + str(f)
        
        if len(words_obj.keys())==0:
            break
        tencent_file[key] = read_tencent(f)
        mated_words = []
        for w in words_obj:
            if w in tencent_file[key]:
                #找到的词添加到mated_words中循环完一个文件后删除words_obj中对应的键值。
                mated_words.append(w)
                for m in words_obj[w]:
                    zero_index[m[0]][m[1]] = tencent_file[key][w]
            else:
                continue
        #销毁文件解除内存占用
        del tencent_file[key]
        for dw in mated_words:
            del words_obj[dw]
        print('step:'+str(f))

    #未能找到的词
    empty_vector = [0 for em in range(200)]
    print('empety words:%d'%len(words_obj.keys()))
    #未找到的词用0代替
    for et in words_obj:
        for c in words_obj[et]:
            zero_index[c[0]][c[1]] = empty_vector

    res = np.array(zero_index)
    np.save(save_path,res)
    #test_in:empety words:1313
    #test_out:empety words:2280


