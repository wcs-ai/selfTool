#!/usr/bin/python
# # -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense
import numpy as np
from selfTool.bt import transformer_model



class __Basic_net__(object):
    _UNIFY_FLOAT = tf.float32
    def __init__(self):
        self._info = { 
            "batch":120,
            "start_learn":1e-3,
            "save_path":'data/',
            "unify_float":tf.float32,
            "sequence_length":200,
            "epoch":10,
            "activate_function":self.Swish,
            "unites":[20,50,80,100,60,30,15],
            "beamWidth":1,
            "x":[],
            "y":[]
        }

    @property
    def arg(self):
        return self._info

    @arg.setter
    def arg(self,val):
        self._info[val[0]] = val[1]
    
    def generator_callback(self):
        for a,b in zip(self._info['x'],self._info['y']):
            yield a,b,len(a),len(b)
    #使用tensorflow的data模块来导入数据
    def load_data(self,ge_fn=None,
                    opt=(tf.int32,tf.int32,tf.int32,tf.int32),
                    osp=([None],[None],(),()),
                    paddings=(0,0,0,0)):
        gf = ge_fn if callable(ge_fn) else self.generator_callback
        dataset = tf.data.Dataset.from_generator(generator=gf,
                                                output_types=opt,
                                                output_shapes=osp)
        dataset = dataset.repeat(self._info['epoch'])
        dataset = dataset.padded_batch(self._info['batch'],osp,paddings)
        iter = dataset.make_one_shot_iterator()

        dt = iter.get_next()
        return dt


    def initial(self,sess):
        sess.run(tf.global_variables_initializer())

    #低版本tensorflow没有该激活函数，自己封装的。
    def Swish(self,x,beta=1):
        return x*tf.nn.sigmoid(x*beta)

    #创建，随机生成参数
    def create_weight(self,size,dtype=None,name="weight"):
        dp = dtype or self._UNIFY_FLOAT
        with tf.device('/cpu:0'):
            res = tf.truncated_normal(size,stddev=0.2,mean=1,dtype=dp)
            variable = tf.get_variable(name=name,initializer=res)

        return variable

    def create_bias(self,size,dtype=None,name="bias"):
        dp = dtype or self._UNIFY_FLOAT
        with tf.device('/cpu:0'):
            bias = tf.constant(0.1,shape=size,dtype=dp,name=name)
            variable = tf.get_variable(name=name,initializer=bias)

        return variable

    #全连接层,一般用于最后一层，默认不使用激活函数
    def fully_connect(self,data,dim,fun=None):
        v = tf.contrib.layers.fully_connected(data,dim,fun)
        return v

    #检查模型是否保存，及其迭代次数
    def check_point(self,save_path):
        #若保存，路径下会有一个checkpoint文件
        kpt = tf.train.latest_checkpoint(save_path)
        if kpt!=None:
            ind = kpt.find('-')
            step = int(kpt[ind+1:])
            return (True,step)
        else:
            return (False,0)
    # 优化器
    def optimize(self,loss,learn_rate=0.1,method="ADM"):
        if method=="ADM":
            _optimize = tf.train.AdamOptimizer
        return _optimize(learn_rate).minimize(loss)

    #计算损失值
    def calc_loss(self,labels,logits,back="sum",method="softmax"):
        loss = ''
        if method=="softmax":
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        elif method=="sparse":
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        else:
            loss = tf.pow(tf.subtract(labels,logits),2)

        if back=="mean":
            return tf.reduce_mean(loss)
        else:
            return tf.reduce_sum(loss)

    # 计算精确度
    def calc_accuracy(self,logits,labels):
        logit_max = tf.argmax(logits,1)
        label_max = tf.argmax(labels,1)
        eq = tf.cast(tf.equal(logit_max,label_max),tf.float32)
        return tf.reduce_mean(eq)
    
    def decay_epoch(self,num,epoch,batch,val=0.9):
        # 返回一个合适的衰退学习率步数。
        _a = (num * val) // batch
        return _a * epoch

    

#用于测试时建立的session
def test_session(data,init=False):
    sess = tf.InteractiveSession()
    if init==True:
        sess.run(tf.global_variables_initializer())
    res = sess.run(data)
    return res


#卷积神经网络
class Cnn(__Basic_net__):
    #transmit a list or array that be createrd layers' arguments
    def __init__(self):
        __Basic_net__.__init__(self)
        self._MODEL = "CNN"
        self.op = 'rnn'
        self.ACTIVATE_FUNCTION = self.Swish

    #封装了归一化、激活的卷积操作,数据、卷积核、偏置值、激活函数、是否是训练状态、滑动步长
    def conv2d(self,data,nucel,bias=0,activate_function=tf.nn.relu,training=True,strides=[1,1,1,1],PADDING='SAME'):
        # x = tf.nn.dropout(data,0.9)
        cvd = tf.nn.conv2d(data,nucel,strides=strides,padding=PADDING)
        if bias!=0:
            cvd = tf.nn.bias_add(cvd,bias)

        norm_cvd = batch_norm(cvd,decay=0.9,is_training=training)
        # norm_cvd = cvd
        elu_cvd = norm_cvd if activate_function==None else activate_function(norm_cvd)
        return elu_cvd    



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
    
    # 用于生成多层cnn模型使用的参数。
    def c_cnn_params(self,weight_width,channels,dtype=tf.float32):
        """args:
        weight_width:width of weights every layer.[3,2,3,4,5,...] or [[1,3,5],3,5,...]
        channels:nucel num every layer,1 d :[3,3,5,...]
        """
        layer_num = len(weight_width)
        assert layer_num + 1 == len(channels)

        layer_weights = []
        layer_biases = []

        def multi_nucel(wls,bls,val,ls_ord,channel_ord,power=1):
            if val > 1:
                wls.append([])
                bls.append([])

                half = channels[channel_ord + 1] // 2
                wls[ls_ord].append(self.create_weight(size=[val,1,channels[channel_ord] * power,half],name='weight_a',dtype=dtype))
                wls[ls_ord].append(self.create_weight(size=[1,val,half,channels[channel_ord + 1]],name='weight_b',dtype=dtype))

                bls[ls_ord].append(self.create_bias(size=[half],dtype=dtype,name='biase_a'))
                bls[ls_ord].append(self.create_bias(size=[channels[channel_ord + 1]],dtype=dtype,name='biase_b'))
            else:
                wls.append(self.create_weight([1,1,channels[channel_ord] * power,channels[channel_ord + 1]]))
                bls.append(self.create_bias([channels[channel_ord + 1]]))

        for i,v in enumerate(weight_width):
            with tf.variable_scope('cnn_layer' + str(i)):
                if type(v)==list or type(v)==tuple:
                    _wt = []
                    _bs = []
                    multiple = 1 if i==0 else 3
                    for j,m in enumerate(v):
                        with tf.variable_scope('multi_nucel' + str(j)):
                            multi_nucel(_wt,_bs,m,ls_ord=j,channel_ord=i,power=multiple)
                        
                    layer_weights.append(_wt)
                    layer_biases.append(_bs)
                else:
                    if i==0:
                        _mp = 1
                    elif type(weight_width[i-1])==list or type(weight_width[i-1])==tuple:
                        _mp = 3
                    else:
                        _mp = 1

                    multi_nucel(layer_weights,layer_biases,v,ls_ord=i,channel_ord=i,power=_mp)

        return layer_weights,layer_biases


    def max_pool(self,data,strides=[1,2,2,1],ksize=[1,3,3,1],PADDING="SAME"):
        pool = tf.nn.max_pool(data,ksize=ksize,strides=strides,padding=PADDING)
        return pool

    def avg_pool(self,data,stride=[1,2,2,1],ksize=[1,3,3,1],PADDING='SAME'):
        pool = tf.nn.avg_pool(data,ksize=ksize,strides=stride,padding=PADDING)
        return pool        

    def cnn_layer(self,data,weights,bias,last=False,mp_stride=[1,1,1,1]):
        # 权重个数必须与偏置个数一一对应。
        layers = []
        convol = data

        multi1 = True if type(weights)==list or type(weights)==tuple else False
        multi2 = False
        _function = self._info['activate_function'] if last else None

        if multi1:
            assert len(weights)==len(bias),'number of weight unequal bias'
            for wg,ba in zip(weights,bias):
                multi2 = True if type(wg)==list or type(wg)==tuple else False
                if multi2:
                    convol = data
                    for w,b in zip(wg,ba):
                        convol = self.conv2d(convol,w,b,_function)
                    layers.append(convol)
                    convol = data
                else:
                    convol = self.conv2d(convol,wg,ba,_function)
                    layers.append(convol)
        else:
            convol = self.conv2d(data,weights,bias,_function)
            
        # 每层只有一个池化操作。
        _res = tf.concat(layers,3) if multi2 else convol

        if last:
          return _res
        else:
          return self.max_pool(_res,mp_stride)

    # 数据，，，平均池化滤波器，结果形状，最大池化滑动步长
    def cnn_run(self,data,weights,biase,shape,mp_stride):
        """args:
        data:4d;
        weights:权重列表[[w1,w2],[w1,w2]],2d;
        biase:偏置列表[b1,b2],1d;
        shape:得到最终结果的形状;
        mp_stide:最大池化，化动步长，[[1,2,2,1],[],...]，最后一层用于平均池化。
        """
        convol_arr = []
        lg = len(biase)
        i = 0
        learn_result = data
        is_last = False

        for w,b,ms in zip(weights,biase,mp_stride):
            if i==lg - 1:
                is_last = True
            else:
                is_last = False
            learn_result = self.cnn_layer(learn_result,w,b,is_last,ms)
            i += 1
        # 全局平均池化层
        avg_res = self.avg_pool(data=learn_result,stride=mp_stride[-1],ksize=mp_stride[-1])
        # all_connect = tf.contrib.layers.fully_connected(res_shape,15,tf.nn.sigmoid)
        return tf.reshape(avg_res,shape)




#普通的循环神经网络构建
class Rnn(__Basic_net__):
    def __init__(self):
        __Basic_net__.__init__(self)
        self._MODEL = 'Rnn'

    def multi_cell(self,layers=3,cell_type='GRU'):
        multi = []
        for i in self._info['unites']:
            if cell_type=='GRU':
                multi.append(tf.contrib.rnn.GRUCell(i))
            else:
                multi.append(tf.contrib.rnn.LSTMCell(i))
                
        mcell = tf.contrib.rnn.MultiRNNCell(multi)
        return tf.nn.rnn_cell.DropoutWrapper(mcell,0.9,0.9)

    def rnn_net(self,data,sequence,layers=3,net_type='dynamic',cell_type='GRU'):
        mcell = self.multi_cell(layers,cell_type)
        if net_type=='static':
            result,state = tf.contrib.rnn.static_rnn(mcell,inputs=data,sequence_length=sequence,
                dtype=self._info['unify_float'],initial_state=None)
        else:
            result,state = tf.nn.dynamic_rnn(mcell,inputs=data,sequence_length=sequence,
                dtype=self._info['unify_float'],initial_state=None)
        return result,state

    def twin_rnn(self,data,sequence,net_type='dynamic'):
        bw_cell = self.multi_cell()
        fw_cell = self.multi_cell()

        if net_type=='static':
            result,_bs,_fs = tf.contrib.rnn.stack_bidirectional_rnn([bw_cell],[fw_cell],data,sequence,
                    dtype=self._info['unify_float'])
        else:
            result,_bs,_fs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([bw_cell],[fw_cell],data,sequence,
                    dtype=self._info['unify_float'])
        return result
        

#使用新版接口构建的seq2seq模型
class Seq2seq(__Basic_net__):
    def __init__(self,encoder_input,decoder_input,hidden_size=3,unite=60,inference=False):
        __Basic_net__.__init__(self)
        self._MODEL = "SEQ2SEQ"
        self.enp = encoder_input
        self.dep = decoder_input

        self.INFERENCE = inference
        self.HIDDEN_NUM = hidden_size
        self.CELL_UNITE = unite

        self.encode_result = ''
        self.encode_state = ''

        self.encode_cell = self.cell()
        self.decode_cell = self.cell()
        self.encoder()

    def encoder(self):
        #用动态rnn构建,encode_state的维度为[batchsize,num_units]
        self.encode_result,self.encode_state = tf.nn.dynamic_rnn(self.encode_cell,self.enp,dtype=self._info['unify_float'])

    def decoder(self,seq_length=None,state_batch=10,model="decoder",start_token=None,end_token=1):
        #为decode层构建一个全连接层，得出每个序列后在乘以该全连接层，把最后的维度转为vocab_len而不是unite数.end_token需要int型
        #project_layer = tf.layers.Dense(units=200,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        project_layer = Dense(self.arg['sequence_length'])
        if not self.INFERENCE:
            #seq_len = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            #sequence_length:当前batch中每条数据的序列数量，超出的话会报错：Tried to read from index 20 but array size is: 20
            helper = TrainingHelper(self.dep,seq_length,time_major=False)
        else:
            helper = GreedyEmbeddingHelper(self.dep,start_token,end_token)

        if model=="decoder":
            train_deocde = BasicDecoder(cell=self.decode_cell,helper=helper,output_layer=project_layer,initial_state=self.encode_state)
        else:
            #生成一个二维的状态数据:[batch_size,num_unit]

            if self._info['beamWidth']>1:
                memory = tile_batch(self.encode_state, multiplier=self._info['beamWidth'])
                #print(memory) [4,?,100]，这里的batch_size用beamWidth*batch,以后多批预测时更改。
                decoder_initial_state = self.decode_cell.zero_state(batch_size=self._info['beamWidth'],dtype=tf.float32)
                decoder_initial_state = decoder_initial_state.clone(cell_state=memory)

                #start_token = tf.ones([self._info['beamWidth']], dtype=tf.int32) * start_token
                #start_token = tf.fill([1],end_token)

                train_deocde = BeamSearchDecoder(cell=self.decode_cell,
                                        embedding=self.dep,
                                        start_tokens=tf.fill([1], 0),
                                        end_token=end_token,
                                        initial_state=decoder_initial_state,
                                        beam_width=self._info['beamWidth'], 
                                        output_layer=project_layer,
                                        length_penalty_weight=1.0)
            else:
                state = self.decode_cell.zero_state(batch_size=state_batch,dtype=tf.float32)
                train_deocde = BasicDecoder(cell=self.decode_cell,
                                            helper=helper,
                                            output_layer=project_layer,
                                            initial_state=state)

        #final_sequence_lengths是一个一维数组，每一条数据的序列数量。output_time_major为False时输出是[batch,seq_num,dim]
        logits,final_state,final_sequence_lengths = dynamic_decode(train_deocde,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=tf.reduce_max(seq_length))
        return logits,final_state,final_sequence_lengths

    def attention_decoder(self,encode_seq_num,state_batch,decode_seq_num=None,start_token=None):
        #num_units与cell中的num_units一致，用于一个全连接权重的列数,与encode层的cell的num_units一样大小。

        if self._info['beamWidth']>1:
            self.encode_result = tile_batch(self.encode_result, multiplier=self._info['beamWidth'])
            encode_seq_num = tile_batch(encode_seq_num, multiplier=self._info['beamWidth'])

        attention_mechanism = LuongAttention(num_units=self.CELL_UNITE,memory=self.encode_result,memory_sequence_length=encode_seq_num)
        #alignment_history表示每一步的alignment是否存储到state中，tenrsorbord可视化时可用,
        #cell_input_fn为一个函数，默认将input和上一步的attention拼接起来送入cell
        self.decode_cell = AttentionWrapper(cell=self.decode_cell,attention_mechanism=attention_mechanism,
            attention_layer_size=None,alignment_history=False)

        logits,final_state,final_sequence_lengths = self.decoder(decode_seq_num,state_batch,
                        model="attention_decoder",start_token=start_token)
        return logits,final_state,final_sequence_lengths
 
    def cell(self):
        cells = []
        for i in range(self.HIDDEN_NUM):
            cells.append(tf.contrib.rnn.GRUCell(self.CELL_UNITE))
        mcell = tf.contrib.rnn.MultiRNNCell(cells)
        return mcell



# 检索式多轮对话模型   
class DAM(__Basic_net__):
    def __init__(self,hp):
        """arg:
        hp:a classa
        """
        __Basic_net__.__init__(self)
        self._MODEL = "DAM"
        self._words_id = {}
        self._trans = transformer_model.Transformer(hp)
        self._cnn = Cnn()

    @property
    def token_id(self):
        return self._words_id
    
    @token_id.setter
    def token_id(self,val):
        self._words_id = val

    # 第一部分：transformer特征提取。
    def feature(self,ids, training=True):
        # ids:2d
        masks = tf.equal(ids,self._words_id['<pad>'])
        embedd = self._trans.embedding_lookup(ids)

        encode_output,src_masks = self._trans.encode(embedd,masks,training)
        return encode_output
    
    # 第二部分：匹配。
    def matching(self,u_sentence,r_sentence):
        """args:
        u_sentence:[batch,sentence_num,seq_len,dim]
        r_sentence:[batch,1,seq_len,dim]
        """
        assert len(u_sentence.get_shape())==4
        assert len(r_sentence.get_shape())==4

        cross = tf.matmul(u_sentence,r_sentence)
        return cross
        
    
    # 第三部分：聚合。
    def aggregation(self,matches,weights,biase,ksize,shape,mp_stride):
        # shape:[batch,sentence_num,seq_len,dim] = > [batch,seq_len,dim,sentence_num]
        sentence_concat = tf.transpose(matches,[0,2,3,1])
        # 多层卷积池化。
        cnn_res = self._cnn.cnn_run(sentence_concat,weights,biase,shape,mp_stride)

        return cnn_res
        
    # union model
    def union_model(self,us,rs,ys,weights,biase,ksize,shape,mp_stride):
        """args:
        us:4d,[batch,sentence_num,seq_len];
        rs:3d,[batch,seq_len];
        ys:label,1d
        """
        batch = us.get_shape().as_list()[0]

        batch_us = []
        for i in range(batch):
            batch_us.append(self.feature(us[i]))
        batch_rs = self.feature(rs)
        us = tf.concat(batch_us,0)

        cross_res = self.matching(us,batch_rs)

        logits = self.aggregation(cross_res,weights,biase,ksize,shape,mp_stride)
        loss = self.calc_loss(labels=ys,logits=logits,back='mean')

        accuracy = self.calc_accuracy(logits=logits,labels=ys)

        rate_step = tf.Variable(0,trainable=False)
        rate = tf.train.exponential_decay(0.05,rate_step,20000,0.1)
        add_step = rate_step.assign_add(1)

        optimize = self.optimize(loss,rate,method='ADM')

        return loss,optimize,rate,add_step





#该类用于和非该文件的深度学习模型结合
class Net(__Basic_net__):
    def __init__(self):
        __Basic_net__.__init__(self)
        self._MODEL = "Basic"

