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
        look_loss = tf.reduce_mean(ce)

        return loss,look_loss

    def train(self, xs, ys,learn_rate=None):
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
        en_masks = tf.equal(xs, self.token2idx['<pad>'])
        xs_enc = self.embedding_lookup(ids=xs,embedding=self.cn_embeddings)

        memory,src_masks = self.encode(xs_enc,en_masks)

        y_input,y_target = ys
        de_masks = tf.equal(y_target, self.token2idx['<pad>'])
        ys_enc = self.embedding_lookup(ids=y_input,embedding=self.cn_embeddings)
        logits, preds = self.decode(ys_enc, memory, src_masks,de_masks)

        
        total_loss,mean_loss = self.calc_loss(y_target,logits)

        global_step = tf.train.get_or_create_global_step()

        # noam_scheme()函数内封装的是自定义的学习率衰减公式，未知
        if learn_rate==None:
            lr = tf.train.polynomial_decay(learning_rate=self.hp.lr,
                                            global_step=global_step,
                                            decay_steps=self.hp.warmup_steps,
                                            end_learning_rate=0.001)
        else:
            lr = learn_rate

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss,global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", total_loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()
        #返回损失值、梯度、步数、训练记录
        return mean_loss,train_op, global_step, lr

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

