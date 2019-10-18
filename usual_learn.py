#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB,ComplementNB
from sklearn.externals import joblib

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


		 