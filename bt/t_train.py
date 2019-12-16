#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from selfTool import deep_learn as dl
from hparams import Hparams
from model import Transformer
from selfTool import file as fl
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

m = Transformer(hp)
with tf.device('/cpu:0'):
	trs = dl.Net() 
	x = np.load('data/test_en.npy',allow_pickle=True)
	y = np.load('data/test_cn.npy',allow_pickle=True)
	cn_vocabs = fl.op_file(file_path='data/cn_words_id.json',method='read')
	en_vocabs = fl.op_file(file_path='data/en_words_id.json',method='read')
	types = ((tf.int32,tf.int32),(tf.int32,tf.int32,tf.int32))
	shapes = (([None],()),([None],[None],()))
	paddings = ((en_vocabs['<pad>'],0),(cn_vocabs['<pad>'],cn_vocabs['<pad>'],0))

def ge_fn():
    for a,b in zip(x,y):
        encode = a + [en_vocabs['</s>']]
        decode = [cn_vocabs['<s>']] + b + [cn_vocabs['</s>']]
        decode_input = decode[:-1]
        target = decode[1:]
        yield (encode,len(encode)),(decode_input,target,len(target))


def train():
	global types,shapes,paddings

	with tf.device('/cpu:0'):
		xs,ys = trs.load_data(ge_fn=ge_fn,opt=types,osp=shapes,paddings=paddings)
		loss, train_op, global_step, rate = m.train(xs, ys)

	gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.9)
	config = tf.ConfigProto(log_device_placement=True,gpu_options=gpu_options,allow_soft_placement=True)

	saver = tf.train.Saver(max_to_keep=6)
	cord = open('data/cord.txt','a+')

	with tf.Session(config=config) as sess:
		trs.initial(sess)

		point = trs.check_point('data/params')
		if point[0]==True:
			saver.restore(sess,'data/params/translate.cpkt-'+str(point[1]))
		start = 0 if point[1]==0 else (point[1]+1)*1000

		num = hp.num_epochs*(1999998//hp.batch_size)
		for i in range(start,num):
			cost,_op,step = sess.run([loss,train_op,global_step])
			print(cost)
			break

			if i%5==0:
				rt = sess.run(rate)
				info = 'step:{};learn:{};loss:{};\n'.format(i,rt,cost)
				cord.write(info)

			if i%1000==0:
				saver.save(sess,'data/params/translate.cpkt',global_step=i//1000)

		summary_writer=tf.summary.FileWriter('data',sess.graph)

train()
#y_hat, eval_summaries = m.eval(xs, ys)
