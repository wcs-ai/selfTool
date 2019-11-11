#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue
from selfTool import common
import jieba
import json,pickle

class statistics_document(object):
    """docstring for read_file"""
    #sequence:文件队列；typ：读取的是中文还是英文en;coording_num:线程数
    def __init__(self, data,coording_num=2,typ='cn',coding='gb2312'):
        #super(read_file, self).__init__()
        #words_list:每个文件的词放到一组[[a,b,v],[]];all_words_list:所有的词放到一起组一个一维组
        self.all_words_list = []
        self.words_list = []
        self.words_obj = []
        self.all_words_obj = {}
        self.file_arr = data
        self.typ = typ
        self.cd_num = coording_num
        self.cd_typ = coding
    def file_take(self):
        datas = queue.Queue(self.cd_num)
        dt = len(self.file_arr) // self.cd_num
        j = 0
        for i in range(self.cd_num):
            if i<self.cd_num-1:
                datas.put(self.file_arr[j:j+dt])
            else:
                datas.put(self.file_arr[j:])
            j += dt
        #启动线程    
        common.start_thread(self.cd_num)
        #get value from quee (get a value every one)
        for que in range(self.cd_num):
            q_data = datas.get()
            self.file_take(q_data)
            for name in q_data:
                content = open(name,'r').read().replace('\n','')
                #每个文档的词列表，词频
                res = self.word_cut(content)
                obj = self.calc_wordsNum(res)
                #self.words_obj.append(obj)
                #self.words_list.append(res)
                #返回一个可迭代对象
                yield (res,obj)
        common.stop_thread()        

        
    #分词、去除停用词、返回词列表，传入一段文字
    def word_cut(self,text):
        words = ''
        all_words = []
        if self.typ=='cn':
            words = jieba.cut(text)
            for w in words:
                all_words.append(w)
        else:
            all_words = text.split()
        
        #去除停用词    
        all_words = common.stop_word(all_words,typ=self.typ)
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
    #处理输入的文本类型    
    def take_text(self):
        words_list = self.word_cut(self.file_arr)
        words_obj = self.calc_wordsNum(words_list)
        return (words_list,words_obj)        
    #不适用迭代使用一次性获取   
    def get_result(self,is_all=False):
        doc_list = []
        doc_obj = []
        all_obj = {}
        for w in self.file_take():
            if is_all==False:
                doc_list.append(w[0])
                doc_obj.append(w[1])
            else:
                doc_list.extend(w[0])
                #合并所有文档词频信息
                for bj in w[1]:
                    if bj in all_obj:
                        all_obj[bj] += w[1].get(bj)
                    else:
                        all_obj[bj] = w[1].get(bj)       
        if(is_all==False):
            return (doc_list,doc_obj)
        else:    
            return (doc_list,all_obj)

#make json or pickle to save file or read file.
def op_file(file_path,data=None,model='json',method='save',):
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

