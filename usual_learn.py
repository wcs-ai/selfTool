#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB


class bayes(object):
	def __init__(self,algorithm="GNB"):
		self.algorithm = algorithm
		 