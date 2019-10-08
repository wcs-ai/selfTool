#!/usr/bin/python
#-*-coding:UTF-8-*-
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class vector(object):
	"""docstring for vector"""
	def __init__(self, save_path):
		#super(vector, self).__init__()
		self.save_path = save_path
		self.vocab_data = ''
	#train vocabulary's vector and save this model
	def train_vocab(self,fileObj,size=100,window=5,min_count=1,workers=5):
		model = Word2Vec(LineSentence(fileObj),size,window,min_count,workers)
		model.save(self.save_path)

	def load_model(self,path):
		self.vocab_data = Word2Vec.load(path)
	#get vector from target file.arguments:file(model path),data(chines words list)
	def get_vector(self,file,data,typ='vocab',back=0):
		word_vector = []
		if typ=='vocab':
			self.load_model(file)
			for dat in data:
				word_vector.append(self.vocab_data.get(dat,default=back))
		return word_vector


			
	def train_section(self,data):
		print('train')

