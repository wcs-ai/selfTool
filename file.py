#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue
import jieba
import json,pickle
import nltk
import numpy as np
try:
    from selfTool import common as cm
except:
    cm = __import__('common')

class Nlp_data(object):
    """docstring for read_file"""
    #sequence:文件队列；typ：读取的是中文还是英文en;coording_num:线程数
    def __init__(self):
        self.vocab = {}
        self.words_to_id_res = []
        self.words_id = {}
    
    #获取：词转id的结果
    @property
    def converse_res(self):
        return self.words_to_id_res

    @converse_res.setter
    def converse_res(self,val):
        self.words_to_id_res = val      
   
    #分词、去除停用词、返回词列表，传入一段文字
    def word_cut(self,text,typ='cn',file_path='selfTool/cn_stop.txt'):
        words = ''
        all_words = []
        if typ=='cn':
            all_words = list(jieba.cut(text))
        else:
            all_words = nltk.word_tokenize(text)

        #去除停用词    
        all_words = cm.stop_word(all_words,typ=typ,file_path=file_path)
        return all_words
    #统计每个词的词频    
    def calc_wordsNum(self,words_list):
        obj = {}
        for word in words_list:
            if word in obj:
                continue
            else:
                obj[word] = words_list.count(word)
        return obj
    
    #剔除小于指定词频的词
    def rid_words(self,words,min_count=10):
        words['<pad>'] = 9999
        words['<s>'] = 9999
        words['</s>'] = 9999
        words['<unk>'] = 9999
        rid_after = {}
        for s in words:
            if words[s]<min_count:
                continue
            else:
                rid_after[s] = words[s]

        self.vocab = rid_after
        return rid_after

    #根据词频表生成词-id，id-词文件
    def c_wid(self,vocab,create_path=''):
        words = list(vocab.keys())
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

        self.words_id = w_id;
        op_file(create_path+'/words_id.json',data=w_id,method='save')
        op_file(create_path+'/id_words.json',data=id_w,method='save')
    
    #将词转换为id
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

    #去除为空的数据
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



#make json or pickle
def op_file(file_path,data=None,model='json',method='save'):
    res = 'ok'
    if model=='json':
        mh = 'w' if method=='save' else 'r'
        with open(file_path,mh) as js:
            if method=='save':
                json.dump(data,js)
            else:
                res = json.load(js)
    else:
        mh = 'wb' if method=='save' else 'rb'
        with open(file_path,mh) as pk:
            if method=='save':
                pickle.dump(data,pk)
            else:
                res = pickle.load(pk)
    return res

# 获取一个文件夹下的所有文件
def get_all_files(path):
    files = []
    def get_file(path):
        assert os.path.exists(path) == True, 'not found target path'
        assert os.path.isfile(path) == False, "target is't a package"
        bags = os.listdir(path)
        for file in bags:
            _p = os.path.join(path, file)
            if os.path.isfile(_p):
                files.append(_p)
            else:
                get_file(_p)
    get_file(path)
    #返回的是所有文件的路径
    return files

#test all encodeing type to decode a byte
def decode_byte(bt,decode=None):
    encodings = ['utf-8','utf-16','utf-32','ascii','Windows-1254','GBK','gb2312','Base64','hex','BIG5','EUC-KR','cp932']
    test = cm.look_encode(bt) or 'utf-8'

    def _dec(st):
        try:
            rq = bt.decode(st)
            return rq
        except:
            return False

    try:
        res = bt.decode(test)
        return res
    except:
        for i in encodings:
            td = _dec(i)
            if td!=False:
                return td
                break
        return False

#读取大型文件时获取指定行
def getLine(path,method='r',ed='utf-8',ord=0):
    i = 0
    file = open(path,method,encoding=ed)
    while i<ord+1:
        line = file.readline()
        if i==ord:
            return line

#把指定位置文件替换到另一个位置
def replace_file(fp='E:\AI\selfTool',tp='E:\AI\selfTool',dt=10):
    import time
    from shutil import copy

    T = time.time()

    #查看指定文件最后修改时间与当前时间差是否小于指定时间
    def in_dt(target_f):
        amend_t = os.stat(nf).st_mtime
        gap_time = (T - amend_t)/60
        return True if gap_time<dt else False

    #直接指定文件的情况，fp，tp都是文件
    if os.path.isfile(fp)==True:
        copy(fp,tp)
    else:
        #fp，tp都是文件夹路径
        files = os.listdir(fp)
        all_file = []
        for f in files:
            nf = os.path.join(fp,f)
            if os.path.isfile(nf) and in_dt(nf):
                copy(nf,os.path.join(tp,f))
            else:
                continue

#replace_file(fp=r'E:\AI\selfTool',tp=r'D:\develope\Anaconda\envs\wcs\Lib\site-packages\selfTool')



