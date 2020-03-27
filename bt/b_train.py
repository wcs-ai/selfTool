#train
import tensorflow as tf
import bert_modeling
import numpy as np
from selfTool import deep_learn as dl
from selfTool import file,data

net = dl.Net()
q = np.load('question.npy',allow_pickle=True)
a = np.load('answer.npy',allow_pickle=True)
vocab = file.op_file('words_id.json',method='read')
types = ((tf.int32,tf.int32),(tf.int32,tf.int32,tf.int32))
shapes = (([None],()),([None],[None],()))
paddings = ((vocab['<pad>'],0),(vocab['<pad>'],vocab['<pad>'],0))

las = [len(i) for i in a]
lqs = [len(j) for j in q]
#print(max(las),max(lqs))#667,1063

def ge_fn():
  global q,a,vocab
  add = lambda s: vocab['<s>'] + s + vocab['</s>']
  for i,j in zip(q,a):
    ei = add(i)
    dj = add(j)

    yield (ei,len(ei)),(dj,len(dj))


def train():
  global types,shapes,paddings,q,a
  #xs,ys = net.load_data(ge_fn=ge_fn,opt=types,osp=shapes,paddings=paddings)
  #xs,ys,x_len,y_len = data.rnn_batch((q,a),batch=10)

  #px,py = data.padding(xs,seq_num=1063,pad=vocab['<pad>'])
  input_x = tf.placeholder(tf.int32,[None,1063])

  config = modeling.BertConfig(
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
      
  #input_ids要求是张量
  model = modeling.BertModel(
          config=config,
          is_training=True,
          input_ids=input_x,
          input_mask=None,
          token_type_ids=None,
          scope=None)
   
  output = model.get_sequence_output()
  vs = tf.contrib.framework.get_variables_to_restore()
  can_restore = [c for c in vs if c.name!='bert/embeddings/word_embeddings:0']

  print([i.name for i in vs])
  return
  saver = tf.train.Saver(can_restore)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ck = net.check_point('bert_ckpt')
    print(ck)
    saver.restore(sess,'bert_ckpt/bert_model.ckpt')
    """
    for xs,ys,x_len,y_len in data.rnn_batch((q,a),batch=10):
      px = data.padding(xs,seq_num=1063,pad=vocab['<pad>'])
      py = data.padding(ys,seq_num=1063,pad=vocab['<pad>'])

      dt = sess.run(output,feed_dict={input_x:px})
      print(np.shape(dt))
      break
    """



train()

def take_data():
    vocab = file.op_file('chate_vocab.json',method='read')

    s = file.Nlp_data()

    s.c_wid(vocab)

    #print(len(ride_vocab.keys()))#13530

def convert_id():
  s = file.Nlp_data()
  x = np.load('chate_data.npy',allow_pickle=True)
  vocab = file.op_file('words_id.json',method='read')
  
  q = []
  a = []
  for i in x:
    qd = i['m'].split()
    ad = i['r'].split()
    q.append(s.word_to_id(qd,vocab=vocab))
    a.append(s.word_to_id(ad,vocab=vocab))
  print(len(q))

  tx,ty = s.drop_empty(q,a)
  print(len(tx))
  np.save('question',tx)
  np.save('answer',ty)


