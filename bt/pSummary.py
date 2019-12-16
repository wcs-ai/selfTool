#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
实现《Pretraining-Based Natural Language Generation for Text Summarization》论文
所述的文本摘要模型
"""
import numpy as np
import tensorflow as tf
from selfTool.bt import t_model,t_modules,b_modeling
import random

class BtSummary(object):
    #config:bert模型的参数配置，hp：transformer模型的参数配置
    def __init__(self,bert_config,tr_config):
        self.MODEL = 'summary'
        #config arguments
        self._bt = bert_config
        self._tr = tr_config

        self.bert_config = b_modeling.BertConfig(
          vocab_size=36200,#36200
          hidden_size=768,#输出的最后一维长度,需要是num_hidden_layers的倍数
          num_hidden_layers=12,
          num_attention_heads=12,
          intermediate_size=3072,#encode端第一层全连接层输出的维度
          hidden_act="gelu",
          hidden_dropout_prob=0.1,
          attention_probs_dropout_prob=0.1,
          max_position_embeddings=512,#与seq_len一致 1063 
          type_vocab_size=2, #生成语句向量时中间使用的过渡数据的维度
          initializer_range=0.02)
        
        self.bModel = b_modeling.BertModel(
          config=self.bert_config,
          is_training=True,
          input_ids=self._dt['bert_input'],
          input_mask=None,
          token_type_ids=None,
          scope=None)
        
        self.tModel = t_model.transformer(tr_config)

    
    @property
    def config(self):
        return self._bt
    @config.setter
    def config(self,val):
        self._bt[val[0]] = val[1]

    #unify embedding provide to transformer and bert.   vocab_embed:user-defined
    def embedding(self,vocab_embed=None):
        embed = vocab_embed or tf.random_uniform([self.bert_config['vocab_size'],self.bert_config['max_position_embeddings']])
        self._vocab_embed = tf.Variable(embed,dtype=tf.float32,name='vocab_embedding')

    #生成每个batch数据的embedding
    def input_embedding(self,input_ids):
      input_embed = self.bModel.create_embedding(input_ids)
      return input_embed
       
    #run bert model
    def bert(self,input_ids):
        self.model._vocabEmbed = self._vocab_embed
        input_embed = self.input_embedding(input_ids)
        self.bModel.calc_output(input_ids,input_embed)
        output = self.bModel.get_sequence_output()
        return output

    #make transformer model
    def transformer(self,encode_output,decode_input,encode_mask,decode_mask):
      """args:
      encode_output:bert模型的输出，对应transformer模型的encoder层输出部分
      decode_input:ids输入数据
      """
      logits, preds = self.tModel.decode(decode_input, encode_output, encode_mask,decode_mask)

      loss = self.tModel.calc_loss(decode_input,logits)
      return loss,preds
      
    #整个模型的流程
    def union_model(self,encode_ids,decode_ids):
      self.embedding()

      #通过bert模型生成文档向量表示
      encode_mask = tf.equal(encode_ids, 0)
      document_embedding = self.bert(encode_ids)
      #第一个decode生成draft summary
      de_masks = tf.equal(decode_input, 0)
      decode_embed = self.input_embedding(decode_ids)
      loss1,preds1 = self.transformer(encode_output=document_embedding,
                                      decode_input=decode_embed,
                                      encode_mask=encode_mask,
                                      decode_mask=de_mask)

      #对草稿中的每个词进行mask
      pred_shape = preds1.get_shape()
      assert len(pred_shape)==2,"pred's shape must be 2 dim"

      one_zero_arr = np.ones(pred_shape)
      mask_arr = np.zeros(pred_shape)
      
      colum = list(range(pred_shape[1]))
      position_arr = [random.sample(colum,pred_shape[1]//2) for ra in range(pred_shape[0])]
      for a in position_arr:
        for b in a:
          one_zero_arr[a][b] = 0
          mask_arr[a][b] = self.tModel.words_id['<mask>']
      
      one_zero_arr = tf.constant(one_zero_arr,dtype=tf.float32)
      mask_arr = tf.constant(mask_arr,dtype=tf.float32)

      mask_ids = preds1*one_zero_arr + mask_arr
      
      #第二次使用bert
      mask_draft_embedding = self.bert(mask_ids)

      #第二个decode
      de_masks2 = tf.equal(preds1, 0)
      loss2,preds2 = self.transformer(encode_output=document_embedding,
                                      decode_input=mask_draft_embedding,
                                      encode_mask=encode_mask,
                                      decode_mask=de_masks2)
      
      #计算两次decode的loss和
      sum_loss = loss1 + loss2
      return sum_loss

    #训练模型
    def fit(self,x,y):
      loss = self.union_model(x,y)
      step = tf.train.get_or_create_global_step()
      lr = t_model.noam_scheme(self._tr.lr, step, self._tr.warmup_steps)
      optimizer = tf.train.AdamOptimizer(lr)
      train_op = optimizer.minimize(loss, global_step=global_step)
      
      return step,lr,loss,train_op
    
    #预测、评估模型
    def predict(self,x):
      pass


