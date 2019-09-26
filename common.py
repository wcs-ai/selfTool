#!/usr/bin/python
#-*-coding:UTF-8-*-
import threading
import queue,os
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
def stop_word(words,typ='cn'):
    #停用词处理，传入数据格式：[(word,v),(),...],纯函数
    
    stop_word_path = 'SELF_TOOLS/stop_words.txt'
    if typ=='cn':
        stop_words = open(stop_word_path,'r').read().split('\n')
    else:
        stop_words = ['__',':',',','.','?','-','!','"','"','(',')']    
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
        if words in stop_words:                     
    return arr
