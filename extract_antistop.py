#!/usr/bin/python
#-*-coding:UTF-8-*-
#extract antistop of article

stop_word_path = 'stop_words.txt'
stop_words = open(stop_word_path,'r').read().split('\n')

def stop_word(words):
    #停用词处理，传入数据格式：[(word,v),(),...],纯函数
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



