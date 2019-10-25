#!/usr/bin/python
# # -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

__UNIFY_FLOAT__ = tf.float32


#创建，随机生成参数
def create_argument(size,dtype=__UNIFY_FLOAT__,name="var"):
        res = tf.random_normal(size,stddev=1,mean=0,dtype=dtype)
        return tf.Variable(res,dtype=dtype)

#封装了归一化、激活的卷积操作,数据、卷积核、偏置值、激活函数、是否是训练状态、滑动步长
def conv2d(data,filter,bas=0,activate_function=tf.nn.relu,training=True,strides=[1,1,1,1],PADDING='SAME'):
    cvd = tf.nn.conv2d(data,filter,strides=strides,padding=PADDING)
    if bas!=0:
        cvd = tf.nn.bias_add(cvd,bas)
    #暂时不添加dropout和正则
    norm_cvd = batch_norm(cvd,decay=0.9,is_training=training)
    elu_cvd = activate_function(norm_cvd)
    return elu_cvd


#低版本tensorflow没有该激活函数，自己封装的。
def Swish(x,beta=1):
    return x*tf.nn.sigmoid(x*beta)

#计算损失值
def calc_loss(labels,logits,method="softmax"):
    loss = ''
    if method=="softmax":
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)

    cost = tf.reduce_mean(loss)
    return cost

#计算精确度
def calc_accuracy(logits,labels):
    res = []
    """
    for i,j in zip(logits,labels):
        a = i if type(i)==np.ndarray else np.array(i)
        b = j if type(j)==np.ndarray else np.array(j)
        if(a==b).all():
            res.append(1)
        else:
            res.append(0)
    """
    eq = tf.equal(logits,labels)
    accu = tf.reduce_mean(tf.cast(eq,tf.int16))
    #accu = np.mean(res)
    return accu

#优化器
def take_optimize(loss,start_learning=0.05,end_step=10,step=10,speed=0.8,method="ADM"):
    rate = tf.train.exponential_decay(start_learning,end_step,step,speed)
    if method=="ADM":
        _optimize = tf.train.AdamOptimizer

    return _optimize(rate).minimize(loss)


#用于测试时建立的session
def test_session(data,init=False):
    sess = tf.InteractiveSession()
    if init==True:
        sess.run(tf.global_variables_initializer())
    res = sess.run(data)
    return res

    
class Cnn(object):
    #transmit a list or array that be createrd layers' arguments
    def __init__(self,wts_size,bas_size):
        assert type(wts_size)==tuple or type(wts_size)==list,"wts_size can't iterable"
        assert type(bas_size)==tuple or type(bas_size)==list,"bas_size can't iterable"

        self.wts = []
        self.bas = []
        self.convole_layers = []
        self.pool_layers = []
        for sw,sb in zip(wts_size,bas_size):
            self.wts.append(create_wt(sw))
            self.bas.append(create_ba(sb))
            self.convole_layers.append(None)
            self.pool_layers.append(None)

    
    def avage_pool(self,img,ksize=[1,3,3,1],stride=[1,2,2,1],PADDING='SAME'):
        pool = tf.nn.avg_pool(img,ksize=ksize,strides=stride,padding=PADDING)
        return pool
    #返回最大池化结果和，最大值位置
    def max_pool(self,img,ksize,stride=[1,2,2,1],PADDING='SAME'):
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
            
    def struct_model(self,img,training):
        image = img
        with tf.device('/gpu:0'):
            for i in range(len(self.bas)):

                self.convole_layers[i] = self.conv2d(image,self.wts[i],self.bas[i],training=training)
                self.pool_layers[i] = self.max_pool(image,[1,2,2,1])
                image = self.convole_layers[i]


class Rnn(object):
    def __init__(self):
        print('rnn')

