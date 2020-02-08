# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf
from selfTool.bt.transformer_modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from tqdm import tqdm
import logging
from selfTool import file as fl

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx = {}
        self.idx2token = {}
        # self.token2idx, self.idx2token = load_vocab(hp.vocab)
        # 不是机器翻译模型时可使用cn相关。
        self.cn_embeddings = get_token_embeddings(hp.cn_vocab_size, hp.d_model, zero_pad=True)
        self.en_embeddings = get_token_embeddings(hp.en_vocab_size, hp.d_model, zero_pad=True)

        # pointer generator network网络部分。
        if hp.pointer_network:
          _vocab_size = hp.cn_vocab_size
          self.hp.cn_vocab_size = hp.d_model
          with tf.variable_scope('decoder'):
            self.w_h = tf.get_variable(name='w_h',shape=[hp.maxlen2,self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.w_s = tf.get_variable(name='w_s',shape=[hp.maxlen2,self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.v = tf.get_variable(name='v',shape=[self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.w_x = tf.get_variable(name='w_x',shape=[hp.maxlen2,self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.w_v = tf.get_variable(name='w_v',shape=[self.hp.cn_vocab_size,_vocab_size],initializer=tf.constant_initializer(0.5))
            self.w_c = tf.get_variable(name='w_c',shape=[hp.maxlen2,self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))

            self.v1 = tf.get_variable(name='va',shape=[hp.maxlen2,self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.b1 = tf.get_variable(name='b1',shape=[self.hp.cn_vocab_size],initializer=tf.constant_initializer(0.5))
            self.b2 = tf.get_variable(name='b2',shape=[_vocab_size],initializer=tf.constant_initializer(0.5))
            self.v2 = tf.get_variable(name='vb',shape=[hp.d_model,_vocab_size],initializer=tf.constant_initializer(0.5))

            self._c = 0
            self._R = 0.5

    @property
    def words_id(self):
      return self.token2idx
    
    @words_id.setter
    def words_id(self,val):
      self.token2idx = val
    
    @property
    def id_words(self):
      return self.idx2token
    
    @id_words.setter
    def id_words(self,val):
      self.idx2token = val

    # data是已经转为词向量的数据
    def take_input(self,data,scope='input',training=True,use_position=True):
        #data是已经转化为词向量的数据。
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # src_masks,比较x中各元素与0比较，为0的变为True否则为False
            #src_masks = tf.equal(data, 0) # (N, T1)
      
            enc = data * self.hp.d_model ** 0.5 # scale

            #positional_encoding内部按照输入的数据的维度生成一个embedding矩阵，目的是加上每个词id的位置id一起作为输入
            if use_position:
              enc += positional_encoding(enc, self.hp.maxlen1)
              enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
        return enc
    
    # 获取词嵌入向量，主要与其它模型结合时提供外部使用。
    def embedding_lookup(self,ids,embedding=None):
        if embedding!=None:
          embedd_arr = embedding
        else:
          embedd_arr = self.cn_embeddings

        embedd = tf.nn.embedding_lookup(embedd_arr,tf.to_int32(ids))
        return embedd

    def encode(self, xs,src_masks, training=True):
        '''
        xs是已经转化为词向量的数据
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        enc = self.take_input(xs,scope='encoder',training=training)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, src_masks

    #ys是已经转化为词向量的数据，tgt_masks:decode输入的masks
    def decode(self, ys, memory, src_masks,tgt_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1),encode输入数据的mask

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        dec = self.take_input(ys,scope='decoder',training=training) 

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward,ff函数中经过两个全连接然后添加残差网络后取ln()结果。
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        if not self.hp.pointer_network:
            # Final linear projection (embedding weights are shared)
            weights = tf.transpose(self.cn_embeddings) # (d_model, vocab_size)
            logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size),矩阵相乘
        else:
            logits = dec
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat

    def calc_loss(self,y,logits,pad=None):
      #不同于0的值为True，转换为0,1矩阵。
        _pad = pad or self.token2idx["<pad>"]
        nonpadding = tf.to_float(tf.not_equal(y, _pad))  # 0: <pad>
        
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.cn_vocab_size)) 
        # train scheme;label_smoothing()函数对生成的one_hot做一个平滑处理，

        #with tf.device('/gpu:0'):
        # 如把1变为0.923,0变为0.034但一个one_hot会保持它最大值得位置不变
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
        #基于权重的交叉熵计算
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        return loss

    def pointer_network(self,encode_out,decode_out,ys):
        # [batch,seq_len,dim]
        e_t = tf.nn.tanh(encode_out * self.w_h + (decode_out * self.w_s) + (self._c * self.w_c))
        a_t = tf.nn.softmax(self.v * e_t)
        # [batch,1,dim]
        h_t = tf.expand_dims(tf.reduce_sum(a_t * encode_out,axis=1),axis=1)
        # [batch,seq_len,dim]
        h_union_s = decode_out + h_t

        # [batch,seq_len,dim]
        p_vocab = h_union_s * self.v1 + self.b1
        # 将长度转为词表长度。[batch,seq_len,vocab_size]
        p_vocab = tf.nn.softmax(tf.einsum('ntd,dk->ntk', p_vocab, self.v2) + self.b2)

        # [batch,seq_len,dim]
        p_gen = h_t * self.w_h + decode_out * self.w_s + ys * self.w_x
        # [batch,seq_len,vocab_size]
        p_gen = tf.nn.sigmoid(tf.einsum('ntd,dk->ntk',p_gen,self.w_v))

        p_w = p_gen * p_vocab + (1 - p_gen) * tf.reduce_sum(a_t)

        self._c = tf.reduce_sum(a_t,axis=1)
        _sc = tf.reduce_sum(self._c)
        _sat = tf.reduce_sum(a_t)
        arr = [_sc,_sat]

        cs = tf.cast([tf.less(_sc,_sat),tf.less(_sat,_sc)],dtype=tf.float32)
        covloss = tf.reduce_sum(arr * cs)

        return p_w,covloss

    def train(self, xs, ys,step):
        '''args:
        xs:ids data 2d.
        ys:3d.
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward,训练中返回的sents1无用，只是配合评估时使用，可以去除。
        en_masks = tf.equal(xs, 0)
        xs_enc = self.embedding_lookup(ids=xs,embedding=self.cn_embeddings)

        memory,src_masks = self.encode(xs_enc,en_masks)

        y_input,y_target,y_len = ys
        de_masks = tf.equal(y_target, 0)
        ys_enc = self.embedding_lookup(ids=y_input,embedding=self.cn_embeddings)
        logits, preds = self.decode(ys_enc, memory, src_masks,de_masks)

        # pointnetwork部分
        if self.hp.pointer_network:
            pw,loss1 = self.pointer_network(memory,logits,ys_enc)
            loss2 = self.calc_loss(y_target,pw)
            total_loss = loss2 + self._R * loss1
        else:
            total_loss = self.calc_loss(y_target,logits)

        global_step = tf.train.get_or_create_global_step()

        # noam_scheme()函数内封装的是自定义的学习率衰减公式，未知
        lr = tf.train.exponential_decay(self.hp.lr, step, self.hp.warmup_steps,0.01)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        #返回损失值、梯度、步数、训练记录
        return total_loss, train_op, global_step, lr

    def eval(self, xs, ys,words_id=None):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        token2idx = words_id or self.token2idx
        #输入·的ys可以随意

        #eval阶段decoder_inputs是未知的，只能随机生成,这里的xs[0]是encode的输入(二维)，所以ones的形状是[batch,1]
        #为每一条数据加一个开始符。
        decoder_inputs = tf.ones((tf.shape(xs)[0], 1), tf.int32) * token2idx["<s>"]
        decode_embed = self.embedding_lookup(ids=decoder_inputs,embedding=self.cn_embeddings)

        xs_enc = self.embedding_lookup(ids=xs,embedding=self.cn_embeddings)
        en_masks = tf.equal(xs, token2idx['<pad>'])
        memory, src_masks = self.encode(xs_enc,en_masks,False)

        tgt_mask = tf.equal(decoder_inputs, token2idx['<pad>'])

        logging.info("Inference graph is being built. Please be patient.")

        #利用得到的词预测下一个词,所以循环maxlen2次
        for _ in range(self.hp.maxlen2):
            #y_hat是一个二维的，y和sents已经无效，因为不用再参与loss的计算。
            logits, y_hat = self.decode(decode_embed, memory, src_masks,tgt_mask,False)
            if tf.reduce_sum(y_hat, 1) == token2idx["<pad>"]: break

            ds = tf.concat((decoder_inputs, y_hat), 1)
            decode_embed = self.embedding_lookup(ids=ds,embedding=self.cn_embeddings)
            tgt_mask = tf.equal(ds,token2idx['<pad>'])

        # monitor a random sample,arg:shape,min.max,type
        
        summaries = tf.summary.merge_all()

        return y_hat, summaries

