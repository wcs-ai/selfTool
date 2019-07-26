#!/usr/bin/python
#-*-coding:UTF-8-*-
#extract antistop of article
from collections import Counter
import functools
from gensim import corpora,models
import math
from jieba import analyse

stop_word_path = 'SELF_TOOLS/stop_words.txt'
stop_words = open(stop_word_path,'r').read().split('\n')

def stop_word(words):
    #停用词处理，传入数据格式：[(word,v),(),...],纯函数
    global stop_words
    arr = []
    for i in words:
        if i[0] in stop_words:
            continue
        else:
            arr.append(i)
    return arr

def filtration_word(words,f_words):
    #过滤词，过滤掉不想要的词性，f_words格式：[v,n,adj,...]
    wds = []
    for k in words:
        if k[1] in f_words:
            continue
        else:
            wds.append(k)
    return wds

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

def clear_one(data):
    print(1)

def antistop(words):
    #过滤词、停用词、统计词
    f_words = filtration_word(words,['v','adj','m','t','x'])
    dt = stop_word(f_words)
    data = [x[0] for x in dt]

    return data



class TFIDF(object):
    def __init__(self,words,num):
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
        print(vs[0:20])

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
def extract(words,num):
    res = TFIDF(words,num)
    hh = res.get_tfidf()
    return hh