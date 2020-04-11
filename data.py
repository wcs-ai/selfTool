#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 
"""
这个文件用来做数据预处理
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json,random
import math,re
import copy,tqdm
import jieba
import nltk
from selfTool import file,common

#数据预处理模块
class Dispose(object):
    def __init__(self):
        self.MODEL = 'data-dispose'
        self.data_type = [list,tuple,np.ndarray,pd.core.frame.DataFrame]
        self.dim = 2
        self.miss_val = [None]
        self.place = 0
    
    # 设置miss值。
    @property
    def miss(self):
        return self.miss_val
    @miss.setter
    def miss(self,val):
        self.miss_val = val

    #查找所有缺失值位置
    def search_miss(self,data):
        miss_index_arr = []
        #允许的数据类型
        for v1,val1 in enumerate(data):
            if type(val1) in self.data_type:
                self.dim = 2
                for v2,val2 in enumerate(val1):
                  #val2是numpy等类型的情况需要变为list
                    try:
                        va2 = list(val2)
                    except:
                        va2 = val2
                    if va2 in self.miss_val:
                        miss_index_arr.append([v1,v2])
            else:
                self.dim = 1
                try:
                    va1 = list(val1)
                except:
                    va1 = val1
                if va1 in self.miss_val:
                    miss_index_arr.append(v1)
        return miss_index_arr

    def _window_data(self,data,idx,window):
        """求出idx附近可用的值
        args:
        data:1d;
        idx:select index;
        window:idx left。idx right。
        """
        val = []
        val.extend(data[idx - window:idx])
        val.extend(data[idx + 1:idx + window])

        # 选到的值及索引。
        save = []
        idxs = []

        for i,v in enumerate(val):
            try:
                ni = list(v)
            except:
                ni = v
            if ni in self.miss_val:
                continue
            else:
                save.append(ni)
                idxs.append(i)
        return save,idxs

    def _mean(self,data):
        """用来计算指定数据的均值。
        
        """
        
        if len(data)==0:
            res = self.place
        else:
            res = np.mean(data, axis=0)
        return res

    
    def interpolate(self,data,axis=1,method="mean",place=0):
        """对给定缺失值类型进行插值.
        args:
        data:2d.
        axis:插值方法选择的指定维度数据作为插值参考。0为选择行。
        method:插值方法.
        place:没有好的插值方案时使用的占位值.
        """
        from scipy.interpolate import lagrange

        self.place = place
        if type(data)==list or type(data)==tuple:
            dt = np.array(data)
        elif type(data)==pd.core.frame.DataFrame:
            dt = np.array(data.values)
        else:
            dt = data
        # 查询数据中的缺失值位置。
        dim = 2 if len(dt.shape)>1 else 1
        miss_idx = self.search_miss(dt)
        
        for val in miss_idx:
            if dim==1:
                send_dt = dt
                pt = val
            else:
                # 判断维度选择参考数据。
                if axis==0:
                    send_dt = dt[val[0]]
                    pt = val[1]
                else:
                    send_dt = dt[:,val[1]]
                    pt = val[0]
            
            _apply_data,idxs = self._window_data(send_dt,pt)
            #判断插值方法,mean:使用缺失值附近的数据的均值代替
            if method=='mean':
                res = self._mean(_apply_data)
            elif method=='lagrange':
                res = lagrange(idxs,_apply_data)(pt)

            #将结果插入数据中
            if dim==1:
                dt[val] = res
            else:
                dt[val[0]][val[1]] = res
        return dt
    
    #数据规范化
    def norm_data(self,data,query_shape,method='norm'):
        shape = np.shape(data)
        take_data = []
        dt = np.reshape(data,query_shape)

        if method=='norm':
            scaler = preprocessing.normalize
        elif method=='max-min':
            scaler = preprocessing.MinMaxScaler()
        elif method=='qt':
            scaler = preprocessing.QuantileTransformer()
        elif method=='mas':
            scaler = preprocessing.MaxAbsScaler()

        #可批量归一化数据
        if len(shape)>2:
            for d in dt:
                if method=='norm':
                    take_data.append(scaler(d,norm='l2'))
                else:
                    take_data.append(scaler.fit_transform(d))
        else:
            d = dt
            if method=='norm':
                take_data = scaler(d,norm='l2')
            else:
                take_data = scaler.fit_transform(d)
        #还原数据形状
        res_data = np.reshape(take_data,shape)
        return res_data


# 专用于处理nlp数据
class Nlp_data(object):
    """docstring for read_file"""
    # sequence:文件队列；typ：读取的是中文还是英文en;coording_num:线程数
    def __init__(self):
        self.vocab = {}
        self.words_to_id_res = []
        self.words_id = {}
        self._vocab_size = 0
    
    @property
    def vocab_size(self):
      return self._vocab_size
    
    @property
    def wid(self):
      return self.words_id
    @wid.setter
    def wid(self,v):
      self.words_id = v

    @property
    def vb(self):
      return self.vocab
    @vb.setter
    def vb(self,val):
      self.vocab = val

    # 获取：词转id的结果
    @property
    def converse_res(self):
        return self.words_to_id_res

    @converse_res.setter
    def converse_res(self,val):
        self.words_to_id_res = val      
   
    # 分词、去除停用词、返回词列表，传入一段文字
    def word_cut(self,text,typ='cn',file_path=r'/home/wcs/item/selfTool/resource/cn_stop.txt',stop=False):
      """args:
      text:一段文字；
      typ:语言类型，en：英文;
      file_path:对应语言的停用词文件位置;
      stop:是否做停用词处理；
      """
      import re

      regs = [re.compile('\n'),re.compile('\s'),re.compile(' ')]
      for rs in regs:
        text = re.sub(rs,'',text)

      all_words = []
      if typ=='cn':
        all_words = list(jieba.cut(text))
      else:
        all_words = nltk.word_tokenize(text)
    
      # 去除停用词
      if stop:
        all_words = common.stop_word(all_words,typ=typ,file_path=file_path)
      return all_words

    # 统计每个词的词频,若需要建立多个词表时需要先清空self.vocab
    def calc_wordsNum(self,words_list):
        for word in words_list:
            if word in self.vocab:
                self.vocab[word] = self.vocab[word] + 1
            elif re.search('\d',word):
                continue
            else:
                self.vocab[word] =  1
    # 直接将整个数据矩阵放入来统计,words:2dim=>[sentence,words]
    def count_all(self,words):
      for w in words:
        self.calc_wordsNum(w)
    
    # 剔除小于指定词频的词
    def rid_words(self,vocab=None,min_count=10):
        words = vocab or self.vocab
        words['<pad>'] = 9999
        words['<s>'] = 9999
        words['</s>'] = 9999
        words['<unk>'] = 9999
        words['<mask>'] = 9999
        rid_after = {w:words[w] for w in words if words[w]>=min_count}

        self._vocab_size = len(rid_after.keys())
        self.vocab = rid_after
        return rid_after

    # 根据词频表生成词-id，id-词文件
    def c_wid(self,vocab=None,create_path='',save=False):
        _vocab = vocab or self.vocab
        words = list(_vocab.keys())
        vocab_len = len(words)
        idxs = list(range(vocab_len))

        w_id = dict(zip(words,idxs))
        id_w = dict(zip(idxs,words))

        #<pad>符号位置 置0，符合习惯。
        _last = id_w[0]
        w_id[_last] = w_id['<pad>']
        id_w[w_id['<pad>']] = _last
        w_id['<pad>'] = 0
        id_w[0] = '<pad>'

        self.words_id = w_id
        print('vocab_size:',len(w_id.keys()))
        if save:
          file.op_file(create_path+'/words_id.json',data=w_id,method='save')
          file.op_file(create_path+'/id_words.json',data=id_w,method='save')
        
        return w_id,id_w

    
    # 将词转换为id,在循环中使用时让back=True，words_llist为1d,back为False时words_list为2d
    def word_to_id(self,words_list,vocab=None,back=True):
        vocab_dict = vocab or self.words_id

        type_list = [list,tuple,np.ndarray]
        _wl = lambda wlst: [vocab_dict.get(w,vocab_dict['<unk>']) for w in wlst]

        if type(words_list[0]) in type_list:
            res = []
            for w1 in words_list:
                if back==False:
                    self.words_to_id_res.append(_wl(w1))
                else:
                    res.append(_wl(w1))
            return res
        else:
            if back==False:
                self.words_to_id_res.append(_wl(words_list))
            else:
                return _wl(words_list)

    # 去除为空的数据
    def drop_empty(self,x,y):
        xd = []
        yd = []
        for a,b in zip(x,y):
            if len(a)==0 or len(b)==0:
                continue
            else:
                xd.append(a)
                yd.append(b)
        return xd,yd
    # 查看最小最大序列长度
    def seq_len(self,ids):
      ls = [len(c) for c in ids]
      return {'min':min(ls),'max':max(ls)}
        


#求两点间距离
def point_distance(x,y):
    a = x if type(x)==np.ndarray else np.array(x)
    b = y if type(y)==np.ndarray else np.array(y)

    c = np.sum(np.square(a-b))
    return np.sqrt(c)


#打乱数据
def shuffle_data(x,y):
    lg = np.arange(0,len(y))
    np.random.shuffle(lg)

    if type(x)!=np.ndarray:
      x = np.array(x)
      y = np.array(y)
    
    res_x = x[lg]
    res_y = y[lg]
    return (res_x,res_y)

#将数据集划分为训练集和测试集
def divide_data(x,y,val=0.8):
    lg = len(y)
    train_len = round(lg*0.8)
    test_len = lg - train_len

    data = {
        "train_x":x[0:train_len],
        "test_x":x[-test_len:],
        "train_y":y[0:train_len],
        "test_y":y[-test_len:]
    }
    return data


#批量读取数据的迭代器。
def get_batch(data,data_num=None,batch=1):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)
    j = 0
    #最后一个迭代项不足batch数时也能使用
    for c in range(iter_numebr):
        batch_res = [i[j:j+batch] for i in data]
        j = j + batch
        yield batch_res

#用于训练rnn网络的数据的特别batch
def rnn_batch(data,batch=1,start_token='',end_token=''):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)

    tp = [list,tuple,np.ndarray]
    #未对齐的数据也能使用
    j = 0
    #最后一个迭代项不足batch数时也能使用 
    for c in range(iter_numebr):
        _x_batch = data[0][j:j+batch]
        _y_batch = data[1][j:j+batch]

        x_batch = [[start_token] + pad + [end_token] for pad in _x_batch]
        y_batch = [[start_token] + pad + [end_token] for pad in _y_batch]

        x_sequence = [np.shape(s)[0] for s in x_batch]
        
        if type(y_batch[0]) in tp:
            y_sequence = [np.shape(s)[0] for s in y_batch]
        else:
            y_sequence = [0 for c in range(batch)]
        
        batch_res = [x_batch,y_batch,x_sequence,y_sequence]

        j = j + batch
        yield batch_res

#填充每条数据的序列数到指定长
def padding(data,seq_num,pad=0):
    """
    data:>=2d
    """
    dt = []

    for i,ct in enumerate(data):
        q = seq_num - len(ct)
        if q >= 0:
            v = [j for j in ct]
            for t in range(q):
                v.append(pad)
            dt.append(v)
        else:
            # 若超出指定长度则截断。
            dt.append(ct[0:seq_num])

    return dt


"""
生成一个和原矩阵一样形状的指定值矩阵，
对未对齐的矩阵使用,只支持2d数据
"""
def fill(data,val=0):
  tp = [list,tuple,np.ndarray]
  res = []
  for i in data:
    n = [val for t in i]
    res.append(n)
  return res


"""用百科词向量将词转为词向量
data:must be 2d,
place:未找到词使用的占位
"""
def baiki_vector(data,baiki_path,module='bin',place=0):
  import gensim
  if module=='bin':
    model = gensim.models.KeyedVectors.load_word2vec_format(baiki_path,binary=True)
  else:
    model = gensim.models.Word2Vec.load(baiki_path)
  
  #存储已查找过的词，重复的则不必再到模型中查找。
  buffer_vector = {}
  res = []

  for sen in data:
    sws = []
    for w in sen:
      if w in buffer_vector:
        sws.append(buffer_vector[w])
      else:
        nk = model[w] if w in model.wv.index2word else place
        sws.append(nk)
        buffer_vector[w] = nk
    sws = [place] if len(sws)==0 else sws
    res.append(sws)
  
  empty_vector = [0 for i in range(128)]
  take = Dispose()
  take.miss = [empty_vector,[],0,None]
  dt = take.interpolate(res,axis=0,method='mean',place=place)

  return dt



# 按照词表id顺序构建一个vector词表，用于embedding_lookup
def embedding_vector(vocab,module='baiki',module_path=None):
    import gensim
    assert module_path!=None,'query module_path of vector file'
    vocab_size = len(vocab.keys())
    vocab_embed = [0 for c in range(vocab_size)]

    if module=='baiki':
        empty = [0 for i in range(128)]
        #model = gensim.models.Word2Vec.load(module_path)
        model = gensim.models.KeyedVectors.load_word2vec_format(module_path,binary=True)
        for key in vocab:
            if key in model.wv.index2word:
                vocab_embed[vocab[key]] = list(model[key])
            else:
                vocab_embed[vocab[key]] = empty
    else:
        pass

    return list(vocab_embed)




#将腾讯的词向量文件分成9个json文件
def divide_tencent_vector(tencent_path):
    file = open(tencent_path,'r',encoding='utf-8')
    save_path = "data/tencent_vector"
    WORDER_NUM = 1000000
    #1000000
    vector = []
    i = 5
    j = 0
    idx = 0

    tencent = {
        "f1":{},
        "f2":{},
        "f3":{},
        "f4":{},
        "f5":{},
        "f6":{},
        "f7":{},
        "f8":{},
        "f9":{},
        "f10":{}
    }

    for item in file:
        if j>WORDER_NUM*4:
            arr = item.split()
            arr_one = arr.pop(0)
            #部分一行有两个中文词
            one = len(arr) - 200
            for n in range(one):
                arr_one = arr.pop(0)

            vector = list(map(float,arr))
                
            idx = "f" + str(i)
            tencent[idx][arr_one] = vector
        else:
            pass
 
        if i < 9:
            if j>WORDER_NUM*i:
                save_name = save_path + str(i) + ".json"
                with open(save_name,'w') as jsonObj:
                    json.dump(tencent[idx],jsonObj)

                    i = i + 1
                del tencent[idx]
            else:
                pass       
        else:
            pass
        j = j + 1

    last_name = save_path + str(i) + ".json"
    with open(last_name,'w') as last:
        json.dump(tencent[idx],last)




def tencent_vector(vocab,tencent_path,save_path='',save=True):
    """根据vocab查找腾讯词向量存为数组，转为embedding_look_up()使用的词向量矩阵。
    args:
    vocab:dict json file;
    tencent_path:腾讯词向量存放的文件夹；
    save_path:保存的文件夹；
    save:是否保存。
    """
    VECTOR_SIZE = 200

    tencent_file = {
        "t1":1,
        "t2":2,
        "t3":3,
        "t4":1,
        "t5":2,
        "t6":3,
        "t7":1,
        "t8":2,
        "t9":3,
    }

    _embedd = [0 for i in range(len(vocab.keys()))]
    _pad = [0 for j in range(VECTOR_SIZE)]
    _unk = [random.uniform(-6,0) for i in range(VECTOR_SIZE)]

    # 读取tencent词向量文件。
    def read_tencent(ord):
        path = tencent_path+'/tc'+ str(ord) +'.json'
        with open(path,'r') as f:
            data = json.load(f)
        return data

    _embedd[vocab['<pad>']] = _pad
    _embedd[vocab['<unk>']] = _pad
    for v in ['<s>','</s>','<mask>']:
        _embedd[vocab[v]] = [random.uniform(-10,-1) for c in range(VECTOR_SIZE)]


    del vocab['<s>'],vocab['</s>'],vocab['<pad>'],vocab['<unk>'],vocab['<mask>']

    #逐个打开腾讯文件
    for f in range(1,10):
        key = 't' + str(f)

        tencent_file[key] = read_tencent(f)
        mated_words = []

        for w in vocab:
            if w in tencent_file[key]:
                mated_words.append(w)
                _embedd[vocab[w]] = tencent_file[key][w]
            else:
                continue
        # 删除已经查找到的词。        
        for w2 in mated_words:
            del vocab[w2]
        #销毁文件解除内存占用
        del tencent_file[key]
    
    # 所有未查找到的用unk填充。
    for c in vocab:
        _embedd[vocab[c]] = _unk

    if save:
        file.op_file(file_path=save_path + '/embedding.pkl',data=_embedd,model='pkl',method='save')
    else:
        return _embedd


