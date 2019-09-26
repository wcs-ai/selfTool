#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue
from SELF_TOOLS import common
import jieba

words_list = []
words_obj = {}
file_words = []
#读取英文文本,参数：文件路径列表
def read_en_txt(sequence,coding):
    global words_list,words_obj
    assert type(sequence)==list,"inputs is't a list"
    coding = coding or 'gb2312'
    q = 0

    en_stop = ['__',':',',','.','?','-','!','"','"','(',')']
    for file in sequence:
        assert os.path.isfile(file)==True,"fileError"
        with open(file,'r') as f:
            try:
                f1 = f.read()
            except:
                print(q)
            for c in en_stop:
                f1 = f1.replace(c,'')
            f2 = f1.lower().replace('<br />','').split()
            file_words.append(f2)
            for word in f2:
                if word in words_list:
                    pass
                else:
                    words_list.append(word)

                if word in words_obj:
                    words_obj[word] += 1
                else:
                    words_obj[word] = 1
        q += 1

#英文读取入口
def en_read(arr,coding,num):
    #添加到队列中做为多线程使用
    datas = queue.Queue(num)
    dt = len(arr) // num
    j = 0
    for i in range(num):
        if i<num+1:
            datas.put(arr[j:j+dt])
        else:
            datas.put(arr[j:])
        j += dt
    common.start_thread(num)
    #get value from quee (get a value every one)
    q_data = datas.get()
    read_en_txt(q_data,coding)

#使用get获取解析的值
def get():
    common.stop_thread()
    return words_list,words_obj,file_words


class statistics_document(object):
    """docstring for read_file"""
    #sequence:文件队列；typ：读取的是中文还是英文en;coording_num:线程数
    def __init__(self, quence,coording_num,typ='cn',coding='gb2312'):
        #super(read_file, self).__init__()
        #words_list:每个文件的词放到一组[[a,b,v],[]];all_words_list:所有的词放到一起组一个一维组
        self.all_words_list = []
        self.words_list = []
        self.words_obj = []
        self.all_words_obj = {}
        self.file_arr = quence
        self.typ = typ
        self.cd_num = coording_num
        self.cd_typ = coding
    def pre_read(self):
        datas = queue.Queue(self.cd_num)
        dt = len(self.file_arr) // self.cd_num
        j = 0
        for i in range(self.cd_num):
            if i<self.cd_num+1:
                datas.put(self.file_arr[j:j+dt])
            else:
                datas.put(self.file_arr[j:])
            j += dt
        common.start_thread(num)
        #get value from quee (get a value every one)
        q_data = datas.get()
        read_en_txt(q_data,coding)
    def cn_take(self):
        

