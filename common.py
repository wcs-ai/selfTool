#!/usr/bin/python
#-*-coding:UTF-8-*-
import threading
import queue,os,chardet
import numpy as np


threads = []
#多线程
class c_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = ''
        self.ord = 0
    def run(self):
        self.ord += 1

def start_thread(num):
    for i in range(num):
        threads.append(c_thread())
        threads[i].start()

def stop_thread():
    for c in threads:
        c.join()


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


#去除停用词,typ:cn为中文，en为英文
def stop_word(words,typ='cn',file_path='SELF_TOOLS/cn_stop.txt'):
    #停用词处理，传入数据格式：[(word,v),(),...],纯函数
    
    # cn_stop = 'SELF_TOOLS/cn_stop.txt'
    # en_stop = 'SELF_TOOLS/en_stop.txt'
    # file_path = cn_stop if typ=='cn' else en_stop
    read = open(file_path,'rb').read()
    code = chardet.detect(read)
    stop_words = read.decode(code['encoding']).replace('\r','').split('\n')    
    arr = []
    res = ''

    #words是一个列表的情况
    if isinstance(words,list) or isinstance(words,tuple):
        for i in words:
            if isinstance(i,list) or isinstance(i,tuple):
                if i[0] in stop_words:
                    continue
                else:
                    arr.append(i)
            else:
                if i in stop_words:
                    continue
                else:
                    arr.append(i)   
    else:
        #传入的是一个字符的情况
        if words in stop_words:
            arr = True
        else:
            arr = False

    return arr

#读取大型文件时获取指定行
def getLine(path,method='r',ed='utf-8',ord=0):
    i = 0
    file = open(path,method,encoding=ed)
    while i<ord+1:
        line = file.readline()
        if i==ord:
            return line

#打乱数据
def shufle_data(x,y):
    lg = np.arange(0,len(y))
    np.random.shuffle(lg)
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
#test the precision about two target
def precision(x,y):
    if isinstance(x,np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y,np.ndarray):
        pass
    else:
        y = np.array(y)

    z = list(x==y)
    leg = len(z)
    ok = z.count(True)
    val = round(ok/leg)
    return val


class _Alone_yield(object):
    def __init__(self,data):
        self.data = data
    def run(self):
        for i in self.data:
            yield i
    def get(self,batch):
        res = []
        i = 0
        for j in self.run():
            if i<batch:
                res.append(j)
            else:
                break
        return res
            

#批量读取数据的迭代器。
def next_batch(data,data_num=None,batch=1):
    data_len = data_num or len(data)
    data_dict = {}
    batch_res = []
    for 
    
    return batch_res

def one_hot(label,deep=10):
    lb = [1 if(c + 1)==label else 0 for c in range(deep)]
    return lb 

#用annoy库对训练好的词向量构建快速查找
def construct_search(path):
    from gensim.models import KeyedVectors
    import json
    from collections import OrderedDict
    tc_wv_model = KeyedVectors.load_word2vec_format(path,binary=True)
    word_index = OrderedDict()
    #counter为索引，key为词;构建一个id词汇映射表，并存为json文件

    for counter,key in enumerate(tc_wv_model.vocab.keys()):
        word_index[key] = counter
        break    

    with open('data/baike.json','w') as fp:
        json.dump(word_index,fp)

    from annoy import AnnoyIndex

    tc_index = AnnoyIndex(128)

    for i,key in enumerate(tc_wv_model.vocab.keys()):
        #tc_wv_model[key]为词对应的词向量
        v = tc_wv_model[key]
        #每条数据按 (索引,词) 加入
        tc_index.add_item(i,v)
        break

    tc_index.build(10)
    tc_index.save('data/baike_index_build.index')

#计算卷积，池化中VALID情况想要的卷积核或滤波器宽度。
def calc_nucleus_width(out_width,input_width,step,method="VALID"):
    if method=="VALID":
        nucleus_width = input_width - (out_width*step - 1)
    return nucleus_width


