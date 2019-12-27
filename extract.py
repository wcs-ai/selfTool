#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
from datetime import datetime
import jieba.posseg as psg
from collections import Counter
import functools
from gensim import corpora,models
import math
import numpy as np
from jieba import analyse
from selfTool import data,common


"""######这里是时间提取部分######"""

h_date = datetime.now()
year = h_date.year
month = h_date.month
day = h_date.day

#春夏秋冬
#分级：年/5,月/4，日/3，时/2，分/1，秒/0
time_words = ['今年','明年','去年','前年','昨天','今天','前天','明天','后天','早上','中午','晚上','傍晚','今早']
reg = {
    'y1':re.compile('[今明去前]年'),
    'y2':re.compile('[0-9零一二两三四五六七八九]+年'),
    'y3':re.compile('年'),
    'm':re.compile('[^0][0-9]?月'),
    'm2':re.compile('月'),
    'd1':re.compile('[昨今前明后](天|日)'),
    'd2':re.compile('[1-3]?[0-9]?[零一二两三四五六七八九十]?(日|号)'),
    'd3':re.compile('日'),
    'h1':re.compile('早上|中午|晚上|傍晚|今早'),
    'h2':re.compile('(\d?\d|[一二两三四五六七八九十]+)(时|点|点钟)'),
    'mint':re.compile("(\d?\d|[一二两三四五六七八九十]+)分"),
    's':re.compile("(\d?\d|[一二两三四五六七八九十]+)秒")
}

def comment_rank(word):
    grade = None
    ys1 = re.search(reg['y1'], word)
    ys2 = re.search(reg['y2'], word)
    ms = re.search(reg['m'], word)
    ds1 = re.search(reg['d1'], word)
    ds2 = re.search(reg['d2'], word)
    hs1 = re.search(reg['h1'], word)
    hs2 = re.search(reg['h2'], word)

    if ys1!=None or ys2!=None:
        grade = 5
    elif ms!=None:
        grade = 4
    elif ds1!=None or ds2!=None:
        grade = 3
    elif hs1!=None or hs2!=None:
        grade = 2

    return grade

#extract time.the arguments:(word,nature of word)
def ext_time(sentence):
    #assert len(data)==2,"data's length must be 2"
    #存储每个时间的数组
    data = []
    word = re.search(reg['s'],sentence)
    st_ = sentence.replace(word.group(),'') if word!=None else sentence
    dat = psg.cut(st_)
    for i,j in dat:
        data.append((i,j))
    
    words = []
    t_arr = []
    
    sentence = ''

    #保留前一个检测到的时间等级，与当前检测到的时间等级若或者大于等于前者新开一个数组值，否则放到前一个组中。
    pre_ga = 6
    #判断是否可添加
    add_sta = False
    for word,nature in data:
        #词性判断
        if nature=='m' or nature== 't':
            sentence += word
        else:
            if sentence!='':
                sentence += word
                words.append(sentence)
            sentence = ''
            continue

    for s in words:
        ts = word_to_time(s)
        t_arr.append(ts)


    return t_arr

def tran_num(sen):
    #点，点钟、分这几个字要去掉
    string = sen
    china_num = ['零','一','二','三','四','五','六','七','八','九','十','时','点','点钟','分','年','月','日']
    a_num = ['0','1','2','3','4','5','6','7','8','9','','','','','','','','']
    for i in range(len(china_num)):
        if china_num[i]!='十':
            string = string.replace(china_num[i],a_num[i])
        else:
            #十的替换比较特殊
            ord = string.find('十')
            if ord==-1:
                continue
    
            if string[ord+1] in china_num and string[ord-1] in china_num:
                string.replace('十','')
            elif string[ord-1] not in china_num and string[ord+1] in china_num:
                string.replace('十','1')
            else:
                string.replace('十','0')

    return string

def word_to_time(sentence):
    global year,month,day
    r_y = ''
    r_m = ''
    r_d = ''
    r_h = ''
    r_min = ''
    #tm1放日期，tm2放具体时间，如果未匹配到具体时间则只返回tm1
    tm1 = []
    tm2 = []

    ys1 = re.search(reg['y1'], sentence)
    ys2 = re.search(reg['y2'], sentence)
    ms = re.search(reg['m'], sentence)
    ds1 = re.search(reg['d1'], sentence)
    ds2 = re.search(reg['d2'], sentence)
    hs2 = re.search(reg['h2'], sentence)
    min1 = re.search(reg['mint'],sentence)

#备注：未对个数年月日处理，如19年，55年等缩写简写。把检测到的带年，月，日等字去掉
    if ys1!=None:
        if(ys1.group()=='前年'):
            r_y = year - 2
        elif ys1.group()=='去年':
            r_y = year - 1
        elif ys1.group()=='明年':
            r_y = year + 1
    elif ys2!=None:
        r_y = tran_num(ys2.group())
    else:
        r_y = year

    #月份检测
    if ms!=None:
        r_m = tran_num(ms.group())
    else:
        r_m = month

    #check day
    if ds1!=None:
        if ds1.group() == '前天':
            r_d = day - 2
        elif ds1.group() == '昨天':
            r_d = day - 1
        elif ds1.group() == '明天':
            r_d = day + 1
        else:
            r_d = day
    elif ds2!=None:
        r_d = tran_num(ds2.group())
    else:
        r_d = day
    tm1.extend([r_y,r_m,r_d])

    #check hour
    if hs2!=None:
        r_h = tran_num(hs2.group())
    else:
        r_h = ''

    if min1!=None:
        r_min = tran_num(min1.group())
    else:
        r_min = '00'    
    tm2.extend([r_h,r_min])
    
    for i in range(len(tm1)):
        tm1[i] = str(tm1[i])

    if r_h=='':
        tm2_string = ''
    else:
        tm2_string = str(tm2[0])+":"+str(tm2[1])       

    return '-'.join(tm1)+","+tm2_string




"""*******这里是关键词提取部分******"""


#对列表中的每个词统计其出现次数返回一个字典
def count_word(word_arr):
    #[word,word,...]
    obj = {}
    res_obj = {}
    for rd in word_arr:
        if rd in obj:
            obj[rd] += 1
        else:
            obj[rd] = 1
#清除只出现一次的词
    for oj in obj:
        if obj[oj]>1:
            res_obj[oj] = obj[oj]
        else:
            continue
    return res_obj



class TFIDF(object):
    def __init__(self,words,num):
        #传入的参数是[[ord,is],[word,row,],...]每一组是一个文档的词，num为关键词数
        self.words = words
        self.sel_num = num
        self.all_words = 0
        self.count_words = {}
        self.tf_dict = {}
        self.idf_dict = {}
        self.TFIDF_DICT = {}
        self.take_data()

    def take_data(self):
        arr = []
        for w in self.words:
            arr += w
        self.count_words = count_word(arr)
        for wd in self.count_words:
            self.all_words += self.count_words[wd]
        self.get_tf()

    def get_tf(self):
        for word in self.count_words:
            self.tf_dict[word] = (self.count_words.get(word,0.0)+1.0)/self.all_words
            self.idf_dict[word] = 0
            for idf in self.words:
                if word in idf:
                    self.idf_dict[word] += 1

    def get_tfidf(self):
        for word in self.count_words:
            self.TFIDF_DICT[word] = self.tf_dict[word]*self.idf_dict[word]


        res =[t[0] for t in sorted(self.TFIDF_DICT.items(),key=lambda k:k[1],reverse=True)]
        return res[0:self.sel_num]

#主题模型
class TopicModel(object):
    #doc_list：多文档列表
    def __init__(self,doc_list,keyword_num,model='LSI',num_topics=4):
        self.dictionary = corpora.Dictionary(doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topics = num_topics

        if model=='LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        #将所有文档词连成一个列表
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf,id2word=self.dictionary,num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf,id2word=self.dictionary,num_topics=self.num_topics)
        return lda

    def word_dictionary(self,doc_list):
        arr = []
        for c in doc_list:
            arr.extend(c)
        return arr

    def get_wordtopic(self,word_dic):
        #词组：[word,word,...]
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            #得到每个词的信息
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    def get_simword(self,word_list):
        #将所有词转为向量表示
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2

            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim
        sim_dic = {}
        for k,v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            #将每个词的信息和文档的信息传入计算各词与文档的相识度。
            sim = calsim(v,senttopic)
            sim_dic[k] = sim
        #sim_dict中每个词的值为0，还存在问题
        vs = [wt[0] for wt in sorted(sim_dic.items(),key=lambda k:k[1],reverse=True)]
        return vs


#调用提取关键词api
def antistop(words_list,keyword_num=10,algorithm='TFIDF'):
    res = ''
    if algorithm=='TFIDF':
        td = TFIDF(words_list,keyword_num)
        res = td.get_tfidf()
    elif algorithm=='LSI':
        topic_model = TopicModel(words_list,keyword_num,model='LSI')
        all_list = []
        for gh in words_list:
            all_list += gh
        res = topic_model.get_simword(all_list)
    elif algorithm=='TEXTRANK':
        #调用textrank算法时直接传入文本数据
        textrank = analyse.textrank
        res = textrank(words_list,keyword_num)
    else:
        topic_model = TopicModel(words_list,keyword_num,model='LDA')
        all_list = []
        for gh in words_list:
            all_list += gh 
        res = topic_model.get_simword(all_list)
    return res    


###***抽取式文本摘要***###
class Summary(object):
    def __init__(self,text,typ='cn'):
      """数据说明：
        text：一个列表，['文档1','文档2',...]多个文档
        self._sentence:源文本句子,[[sentence1,sentence2,...],[sentence]]
        self._txt_words:源文本词,[[[w1,w2],[w1,w2,w3],...],[]]
        self._key_words:所有文档的关键词列表
        self._sentence_vector:所有文档的句向量
        self._sentence_similar:句子相似度矩阵
      """
      self._text = text
      self._typ = typ
      self._divide_alpha1 = '。' if typ=='cn' else '.'
      self._divide_alpha2 = '；' if typ=='cn' else ';'
      self._sentence = []
      self._txt_words = []
      self._vocab_count = {}
      self._words_id = {}
      self._key_words = []
      self._sentence_vector = []
      self._sentence_similar = []

      self.pre_data = data.Nlp_data()
      self.divide_text()
    
    #将多文档分别分句
    def divide_text(self,text=None):
      txt = text or self._text
      sentence1 = [t.split(self._divide_alpha1) for t in txt]
      sentence = []
      for s in sentence1:
        sen = [i.split(self._divide_alpha2) for i in s]
        _sen = [k for j in sen for k in j]
        sentence.append(_sen)
      self._sentence = sentence
    
    #将各文档划分为1个个词
    def divide_word(self):
      tx = []
      for t in self._sentence:
        doc = []
        for s in t:
          doc.append(self.pre_data.word_cut(s,stop=True))
        tx.append(doc)
        self.pre_data.count_all(doc)
      
      #dim:[doc_num,sentence_num,words_num]
      self._txt_words = tx
      #词频字典
      self._vocab_count = self.pre_data.rid_words(min_count=0)
      self._words_id,_idw = self.pre_data.c_wid(save=False)
    
    
    #将词转为id，转为词向量
    def _convert(self):
      #将所有文档联合成一个
      all_doc = self._union(self._txt_words)

      #将整篇文档转换为id表示
      #self.pre_data.word_to_id(all_doc,back=False)
      #ids = self.pre_data.converse_res

      #用腾讯词向量转为词向量
      empty = [0 for i in range(128)]
      vector = data.baiki_vector(all_doc,baiki_path='G:/baike_news.bin',place=empty)

      #生成句子的句向量
      for s in vector:
        self._sentence_vector.append(np.mean(s,axis=0))

    
    #欧式距离计算句子相似度矩阵
    def sentence_similary(self):
      sentence_size = len(self._sentence_vector)
      similary_arr = np.zeros((sentence_size,sentence_size),dtype=np.float32)
      
      #上三角矩阵计算相似度
      for ai,a in enumerate(self._sentence_vector):
        for bi,b in enumerate(self._sentence_vector):
          if bi>ai:
            #计算余弦距离
            similary_arr[ai][bi] = data.point_distance(a,b)

      #副对角线对称实现
      self._sentence_similar = similary_arr + similary_arr.T

    #获取关键词
    def get_key_word(self,algorithm='TFIDF',key_num=100):
      words_list = self._union(self._txt_words)
      input_data = '。'.join(self._text) if algorithm=='TEXTRANK' else words_list
      self._key_words = antistop(input_data,algorithm=algorithm,keyword_num=key_num)

    #将内一层的矩阵连接成一个
    def _union(self,data):
      m = []
      for j in data:
        m = m + j
      return m
    
    #对句子分数排序
    def rank_score(self,score_dict,sentences,num=4):
      sort_score = [si[0] for si in sorted(score_dict.items(),key=lambda k: k[1],reverse=True)]
      res = [sentences[gh] for gh in sort_score[0:num]]
      return self._divide_alpha1.join(res) + self._divide_alpha1
    
    ###***以下是各种算法实现***###----------

    #leader3 to create summary
    def leader3(self):
      us = self._union(self._sentence)
      sumary = self._divide_alpha1.join(us[0:3]) + self._divide_alpha1
      return sumary
    
    #textrank create summary
    def textrank(self,num=5):
      self.divide_word()

      all_doc = self._union(self._txt_words)
      doc_sentence = self._union(self._sentence)
      bm25 = common.BM25(all_doc)
      #各句的初始分数
      INITIAL_SCORE = 1
      D = 0.85

      sentence_len = len(all_doc)
      sentence_score = {ord:INITIAL_SCORE for ord in range(sentence_len)}

      #textrank公式实现
      def iter_score(scores,idx):
        scores.pop(idx)
        weight_sum = sum(scores)
        
        #过滤一些特殊情况
        if weight_sum<=0:
          return 0

        last_score = 1
        #每轮更新last_score的值
        for s1 in scores:
          score = 0
          for s2 in scores:
            score += last_score*s2 / weight_sum
          last_score = 1 - D + D*score
        return last_score

      #计算每个句子的得分
      for i,ic in enumerate(all_doc):
        now_scores = bm25.every_score(ic)
        _score = iter_score(now_scores,i)
        sentence_score[i] = _score

      #按分数从大到小排序
      sumary = self.rank_score(sentence_score,doc_sentence,num=num)
      
      return sumary
    
    #基于关键词的文本摘要生成
    def base_keyword(self,num=4):
      self.divide_word()
      self.get_key_word()
      words_list = self._union(self._txt_words)
      doc_sentence = self._union(self._sentence)

      #初始化
      INITIAL_SCORE = 0
      CLUSTER_GAP = 5
      SCORE_SMOOTH = 0.7
      SUITABLE_LEN = 20
      sentence_score = {s:INITIAL_SCORE for s in range(len(doc_sentence))}

      #记录每句中关键词的位置
      keywords_score = {w:(len(self._key_words)+1-sc) for sc,w in enumerate(self._key_words)}
      
      #一个倒立的，最大值在第一象限的抛物线公式。句子长度在SUITABLE_LEN左右时有较高的得分
      len_score = lambda x: -(x-SUITABLE_LEN)**2 + 50
      #输入句子，返回每句所有关键词总分，和位置
      def kp(ws):
        position_arr = []
        score = 0
        for ix,w in enumerate(ws):
          if w in self._key_words:
            score += keywords_score[w]*SCORE_SMOOTH
            position_arr.append(ix)
        return position_arr,score
            
      for si,sen in enumerate(words_list):
        key_words_position,sen_key_score = kp(sen)

        if len(key_words_position)==1:
          sentence_score[si] = round(1/len(sen),5)
        else:
          #一句有多个关键词的情况
          start = key_words_position[0] if len(key_words_position)!=0 else 0
          keys = 1
          for cs in range(1,len(key_words_position)):
            if key_words_position[cs] - key_words_position[cs-1]<CLUSTER_GAP:
              keys += 1
              end = key_words_position[cs]
            else:  
              """
              当前关键词位置与前一个关键词不构成一个簇的情况时,开始计算上一个簇分数.
              只有一个关键词时分数都一样：1/CLUSTER_GAP
              """
              score = 1/CLUSTER_GAP if keys==1 else round(keys**2/(end-start),5)
              sentence_score[si] += score
              #更新start位置为当前位置
              start = key_words_position[cs]
              keys = 1
          #对最后一个情况计算得分
          score = 1/CLUSTER_GAP if keys==1 else round(keys**2/(end-start),5)
          sentence_score[si] += (score + sen_key_score) / (len_score(len(sen)) + 1)
      
      #排序
      sumary = self.rank_score(sentence_score,doc_sentence,num)
      return sumary

    #基于k-means的文本摘要生成
    def km(self,num=5):
      from sklearn.cluster import KMeans
      self.divide_word()
      self._convert()

      doc_sentence = self._union(self._sentence)
      kmeans = KMeans(init="k-means++",n_clusters=num)
      kmeans.fit_predict(self._sentence_vector)

      #将各句子索引分到不同的类中
      ct = {s:[] for s in range(num)}
      for ic,c in enumerate(kmeans.labels_):
        ct[c].append(ic)
      
      ct_score = []
      for h in ct:
        dist_arr = {}
        for j in ct[h]:
          score = data.point_distance(self._sentence_vector[j],kmeans.cluster_centers_[h])
          dist_arr[j] = score
        ct_score.append(common.rank_dict(dist_arr,reverse=False)[0])

      res = [doc_sentence[idx] for idx in ct_score]
      return res
      

