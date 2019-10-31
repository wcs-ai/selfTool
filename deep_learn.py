#!/usr/bin/python
# # -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

__UNIFY_FLOAT__ = tf.float32

class __Basic_net__(object):
    def __init__(self):
        self.BATCH_SIZE = 300
        self.START_LEARN_RATE = 0.1
        self.TRAIN_CORD = {}
        self.SAVE_PATH = 'data/model.cpkt'
        self.CONVOL = 1


#创建，随机生成参数
def create_weight(size,dtype=__UNIFY_FLOAT__,name="weight"):
    res = tf.truncated_normal(size,stddev=1,mean=0,dtype=dtype)
    return tf.Variable(res,dtype=dtype)
def create_bias(size,dtype=__UNIFY_FLOAT__,name="bias"):
    bias = tf.constant(0.1,shape=size,dtype=dtype,name=name)
    return tf.Variable(bias,dtype=dtype,name=name)

#封装了归一化、激活的卷积操作,数据、卷积核、偏置值、激活函数、是否是训练状态、滑动步长
def conv2d(data,nucel,bias=0,activate_function=tf.nn.relu,training=True,strides=[1,1,1,1],PADDING='SAME'):
    #x = tf.nn.dropout(data,0.8)
    print(nucel)
    cvd = tf.nn.conv2d(data,nucel,strides=strides,padding=PADDING)
    if bias!=0:
        cvd = tf.nn.bias_add(cvd,bias)

    norm_cvd = batch_norm(cvd,decay=0.9,is_training=training)
    #norm_cvd = cvd
    elu_cvd = activate_function(norm_cvd)
    return elu_cvd


#低版本tensorflow没有该激活函数，自己封装的。
def Swish(x,beta=1):
    return x*tf.nn.sigmoid(x*beta)

#计算损失值
def calc_loss(labels,logits,back="mean",method="softmax"):
    loss = ''
    if method=="softmax":
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)

    if back=="mean":
        return tf.reduce_mean(loss)
    else:
        return tf.reduce_sum(loss)

#计算精确度
def calc_accuracy(logits,labels):

    logit_max = tf.argmax(logits,1)
    label_max = tf.argmax(labels,1)
    eq = tf.cast(tf.equal(logit_max,label_max),tf.float32)
    return tf.reduce_mean(eq)

#优化器
def take_optimize(loss,learn_rate=0.1,method="ADM"):
    if method=="ADM":
        _optimize = tf.train.AdamOptimizer

    return _optimize(learn_rate).minimize(loss)

#用于测试时建立的session
def test_session(data,init=False):
    sess = tf.InteractiveSession()
    if init==True:
        sess.run(tf.global_variables_initializer())
    res = sess.run(data)
    return res

def max_pool(data,ksize=[1,3,3,1],strides=[1,2,2,1]):
    pool = tf.nn.max_pool(data,ksize=ksize,strides=strides,padding="SAME")
    return pool

def avg_pool(data,ksize=[1,3,3,1],stride=[1,2,2,1],PADDING='SAME'):
    pool = tf.nn.avg_pool(data,ksize=ksize,strides=stride,padding=PADDING)
    return pool

#检查模型是否保存，及其迭代次数
def check_point(save_path):
    #若保存，路径下会有一个checkpoint文件
    kpt = tf.train.latest_checkpoint(save_path)
    if kpt!=None:
        ind = kpt.find('-')
        step = int(kpt[ind+1:])
        return (True,step)
    else:
        return (False,0)

class Cnn(object):
    #transmit a list or array that be createrd layers' arguments
    def __init__(self):
        self.op = 'rnn'
        self.ACTIVATE_FUNCTION = Swish

    def multi_layer(self,data,weights,biass):
        argument_num = len(biass)
        layer_res = data
        i = 0
        for w,b in zip(weights,biass):
            if i==argument_num-1:
                layer_res = conv2d(layer_res,w,b,activate_function=self.ACTIVATE_FUNCTION)
            else:
                cvd = conv2d(layer_res,w,b)
                layer_res = max_pool(cvd)
        return layer_res


    #返回最大池化结果和，最大值位置
    def max_pool_mask(self,img,ksize,stride=[1,2,2,1],PADDING='SAME'):
        _a,mask = tf.nn.max_pool_with_argmax(img,ksize=ksize,strides=stride,padding=PADDING)
        mask = tf.stop_gradient(mask)
        res = tf.nn.max_pool(img,ksize=ksize,strides=stride,padding=PADDING)
        return res,mask

    #反最大池化与反平均池化
    def unpool(self,tp,fw,step=None,padding="SAME"):
        #默认滑动步长为2
        slide_step = 2 if step==None else step
        shape = tp.shape if type(tp)==np.ndarray else np.shape(tp)
        assert len(shape)==3,"tp's shape must be 3"
        for i in range(shape[2]):
            pass
        if tp=='max':
            print(1)
        else:
            pass

    def run(self,data,weights,biass,ksize,shape):
        last_res = self.multi_layer(data,weights,biass)
        avg = avg_pool(last_res,ksize,stride=ksize)
        return tf.reshape(avg,shape)
        #return last_res


class Rnn(object):
    def __init__(self):
        print('rnn')

