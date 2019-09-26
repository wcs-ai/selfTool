#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
from datetime import datetime
import jieba.posseg as psg
from collections import Counter
import functools
from gensim import corpora,models
import math
from jieba import analyse


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

"""
def antistop(words):
    #过滤词、停用词、统计词
    f_words = filtration_word(words,['v','adj','m','t','x'])
    dt = stop_word(f_words)
    data = [x[0] for x in dt]

    return data
"""


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

        vs = [wt[0] for wt in sorted(sim_dic.items(),key=lambda k:k[1],reverse=True)]
        #print(vs[0:20])

#主题模型提取关键词
def topic_extract(word_list,pos=False,keyword_num=10):
    topic_model = TopicModel(word_list,keyword_num)
    all_list = []
    for gh in word_list:
        all_list += gh
    print(all_list)
    topic_model.get_simword(all_list)

#texrank算法提取关键词,直接输入文本
def textrank_extract(text,pos=False,keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text,keyword_num)
    for keyword in keywords:
        print(keyword+'/')

#TFIDF
def extract_tfidf(words,num):
    res = TFIDF(words,num)
    hh = res.get_tfidf()
    return hh


#调用提取关键词api
def antistop(words_list,keyword_num=10,algorithm='TFIDF'):
    res = ''
    if algorithm=='TFIDF':
        td = TFIDF(word_list,keyword_num)
        res = td.get_tfidf()
    elif algorithm=='LSI':
        topic_model = TopicModel(word_list,keyword_num,model='LSI')
        all_list = []
        for gh in word_list:
            all_list += gh
        res = topic_model.get_simword(all_list)
    elif algorithm=='TEXTRANK':
        #调用textrank算法时直接传入文本数据
        textrank = analyse.textrank
        res = textrank(words_list,keyword_num)
    else:
        topic_model = TopicModel(word_list,keyword_num,model='LDA')
        all_list = []
        for gh in word_list:
            all_list += gh
        res = topic_model.get_simword(all_list)