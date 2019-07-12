#!/usr/bin/python
#-*-coding:UTF-8-*-
#extract antistop of article
from collections import Counter
import functools

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

def extract(words,num):
    res = TFIDF(words,num)
    hh = res.get_tfidf()
    return hh








