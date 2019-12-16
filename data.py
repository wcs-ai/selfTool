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
import math
import copy

class Dispose(object):
    def __init__(self):
        self.MODEL = 'data-dispose'
        self.data_type = [list,tuple,np.ndarray,pd.core.frame.DataFrame]
        self.dim = 2
        self.miss_val = [None]
        self.place = 0
    @property
    def miss(self):
        return self.miss_val
    @miss.setter
    def miss(self,val):
        self.miss_val = val

    def search_miss(self,data):
        miss_index_arr = []
        #允许的数据类型
        for v1,val1 in enumerate(data):
            if type(val1) in self.data_type:
                self.dim = 2
                for v2,val2 in enumerate(val1):
                    if list(val2) in self.miss_val:
                        miss_index_arr.append([v1,v2])
            else:
                self.dim = 1
                if val1 in self.miss_val:
                    miss_index_arr.append(v1)
        return miss_index_arr

    def _mean(self,data,idx,window=3):
        #统一传来的数据都是一维的
        val = []
        val.extend(data[idx - window:idx])
        val.extend(data[idx + 1:idx + window])
        save = []
        for i in val:
            if list(i) in self.miss_val:
                continue
            else:
                save.append(i)
        if len(save)==0:
            res = self.place
        else:
            res = np.mean(save, axis=0)
        return res

    #对给定缺失值类型进行插值，arguments:插值时依据的数据维度、插值方法、没有好的插值方案时使用的占位值
    def interpolate(self,data,axis=1,method="mean",place=None):
        self.place = place
        if type(data)==list or type(data)==tuple:
            dt = np.array(data)
        elif type(data)==pd.core.frame.DataFrame:
            dt = np.array(data.values)
        else:
            dt = data

        miss_idx = self.search_miss(dt)
        for val in miss_idx:
            if self.dim==1:
                send_dt = dt
                pt = val
            else:
                if axis==0:
                    send_dt = dt[val[0]]
                    pt = val[1]
                else:
                    send_dt = dt[:,val[1]]
                    pt = val[0]
            #判断插值方法
            if method=='mean':
                res = self._mean(send_dt,pt)

            #将结果插入数据中
            if self.dim==1:
                dt[val] = res
            else:
                dt[val[0]][val[1]] = res
        return dt


#求两点间距离
def point_distance(x,y):
    a = x if type(x)==np.ndarray else np.array(x)
    b = y if type(y)==np.ndarray else np.array(y)

    c = np.sum(np.square(a-b))
    return np.sqrt(c)


#打乱数据
def shufle_data(x,y):
    lg = np.arange(0,len(y))
    np.random.shuffle(lg)
    res_x = x[lg]
    res_y = y[lg]
    return (res_x,res_y)

#将数据集划分为训练集和测试集
def divide_data(x,y,val=0.8):
    lg = len(y)
    train_len = round(lg*0.8)
    test_len = lg - train_len

    data = {
        "train_x":x[0:train_len],
        "test_x":x[-test_len:],
        "train_y":y[0:train_len],
        "test_y":y[-test_len:]
    }
    return data


#批量读取数据的迭代器。
def get_batch(data,data_num=None,batch=1):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)
    j = 0
    #最后一个迭代项不足batch数时也能使用
    for c in range(iter_numebr):
        batch_res = [i[j:j+batch] for i in data]
        j = j + batch
        yield batch_res

#用于训练rnn网络的数据的特别batch
def rnn_batch(data,batch=1):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)

    tp = [list,tuple,np.ndarray]
    #未对齐的数据也能使用
    j = 0
    #最后一个迭代项不足batch数时也能使用 
    for c in range(iter_numebr):
        x_batch = data[0][j:j+batch]
        y_batch = data[1][j:j+batch]

        x_sequence = [np.shape(s)[0] for s in x_batch]
        
        if type(y_batch[0]) in tp:
            y_sequence = [np.shape(s)[0] for s in y_batch]
        else:
            y_sequence = [0 for c in range(batch)]
        
        batch_res = [x_batch,y_batch,x_sequence,y_sequence]

        j = j + batch
        yield batch_res

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

#填充每条数据的序列数到指定长
def padding(data,seq_num,pad=0):
    datas = copy.deepcopy(list(data))
    dt = []

    for i,ct in enumerate(datas):
        q = seq_num - len(ct)

        assert q>=0,'len(ct):{},that is lenly then seq_num:{}'.format(len(ct),seq_num)
        for c in range(q):
            if type(datas)==np.ndarray:
                np.append(datas[i],pad)
            else:
                datas[i].append(pad)
            
            #np.append(datas[i],emp)
        dt.append(datas[i])
    return dt

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


