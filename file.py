#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue
from SELF_TOOLS import common

words_list = []
words_obj = {}
file_words = []
#读取英文文本,参数：文件路径列表
def read_en_txt(sequence,coding):
    global words_list,words_obj
    assert type(sequence)!='list',"inputs is't a list"
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


def en_read(arr,coding,num):
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
    q_data = datas.get()
    read_en_txt(q_data,coding)

#使用get获取解析的值
def get():
    common.stop_thread()
    return words_list,words_obj,file_words



