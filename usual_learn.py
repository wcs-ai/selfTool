#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB
from sklearn.externals import joblib
from sklearn.cluster import KMeans,MiniBatchKMeans
import json

class bayes(object):
	def __init__(self,data,target,algorithm="GNB"):
		self.algorithm = algorithm
		self.data = data
		self.target = target
		if algorithm=='GNB':
			self.model = GaussianNB()
		elif algorithm=='MNB':
			self.model = MultinomialNB()
		elif algorithm=='BNB':
			self.model = BernoulliNB()
		else:
			self.model = ComplementNB()

		self.model.fit(data,target)

	def save_model(self,path):
		joblib.dump(self.model,path)

	def load_model(self,path):
		self.model = joblib.load(path)

	def predict(self,x):
		res = self.model.predict(x)
		return res

#[9,10,5]
class Layer_kmeans(object):
	def __init__(self,cluster):
		self.MODEL = "Layer_kmeans"
		self._cluster = cluster
		self._clust_len = 0
		self._cluster_tree = {
			"position":'root',
			"festival":[],
			"center_point":None
		}

	@property
	def result(self):
		return self._cluster_tree

	#arguments:the target data(mast be 2d),words with data,
	def tencent(self,data,words,clusters=[5]):
		_kmeans = {}
		_kmeans_tree = {
			"name":"root",
			"center_point":[],
			"festival":[]
		}

		for i,la in enumerate(clusters):
			key1 = 'layer'+str(i)
			key2 = 'vector' + str(i)
			_kmeans[key1] = KMeans(init="k-means++",n_clusters=la)
			_kmeans[key2] = {}

		_kmeans['layer0'].fit_predict(data)
		_kmeans_tree['center_point'] = _kmeans['layer0'].cluster_centers_

		#将所有数据按类分开,存成字典
		for a,b in enumerate(_kmeans['layer0'].labels_):
			key = 'vector' + str(b)
			_kmeans[key][words[a]] = data[a]
		#各类存到不同的文件
		for i in range(clusters[0]):
			k = 'vector' + str(i)
			save_path = 'data/tencent/tree' + str(i) +'.json'
			with open(save_path,'w') as f:
				json.dump(_kmeans[k],f)
			del _kmeans[k]

	#这里开以开多线程操作,info with data
	def cluster(self,data,keys,save_path):
		self._clust_len = len(self._cluster) - 1
		self._basic_cluster(data,keys,self._cluster_tree,0)
		with open(save_path,'w') as res:
			json.dump(self._cluster_tree,res)

	#参数：聚类数据、类数，当前层位置
	def _basic_cluster(self,data,keys,tree_obj,position=0):

		if position=='last':
			n_clusters = self._cluster[self._clust_len]
		else:
			n_clusters = self._cluster[position]

		dts = []
		for v in range(n_clusters):
			dts.append({})

		km = KMeans(init="k-means++",n_clusters=n_clusters)
		km.fit_predict(data)

		#将得到的各类别分开
		for i,j in enumerate(km.labels_): 
			dts[j][keys[i]] = data[i]

		#利用对象传参是按引用传递的方法来完善整颗树。
		tree_obj['center_point'] = km.cluster_centers_
		tree_obj['position'] = position

		if position!='last':
			for i,g in enumerate(dts):
				tree_obj['festival'].append({
						"center_point":0,
						"festival":[],
						"position":''
					})
				pt = 'last' if position==self._clust_len else (position + 1)
				next_keys = list(g.keys())
				next_values = list(g.values())
				self._basic_cluster(next_values,next_keys,tree_obj['festival'][i],pt)
		else:
			#至此一个循环完成
			tree_obj['festival'] = dts



 
		 