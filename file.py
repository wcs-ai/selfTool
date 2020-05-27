#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue
import re
import json,pickle
import numpy as np
from selfTool import common as cm


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
            return rq,st
        except:
            return False

    try:
        res = bt.decode(test)
        return res,test
    except:
        for i in encodings:
            td = _dec(i)
            if td!=False:
                return td
                break
        return False,False

#读取大型文件时获取指定行
def getLine(path,method='r',ed='utf-8',ord=0):
    i = 0
    file = open(path,method,encoding=ed)
    while i<ord+1:
        line = file.readline()
        if i==ord:
            return line

#把指定位置文件替换到另一个位置
def replace_file(fp='E:\AI\selfTool',tp='E:\AI\selfTool',dt=30):
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




class WPE_op(object):
    # 读取pdf,word文档，exe表格等文件。
    def __init__(self):
        self._file = ''
        self._allow_file_type = ['pdf','docx','csv','txt']
        self._read_result = ''

    def read(self,file_path):
        self._file = file_path
        self._judge_file(file_path)

        if self._fileType=='docx':
            self._docx()
        elif self._fileType=='pdf':
            self._pdf()
        elif self._fileType=='csv':
            self._csv()
    
    def save(self,dt,filname='wpe.csv'):
 
        self._judge_file(filname)
        if self._fileType=='csv':
            self.__csv(dt,filname)

    def _judge_file(self,file_path):
        # 判断文件是否存在和文件类型。
        #assert os.path.exists(file_path),'not fount {}'.format(file_path)

        names = os.path.splitext(file_path)
        assert names[1][1:] in self._allow_file_type,"dont be allowed file"

        self._fileType = names[1][1:].lower()
    

    def _re_row(self,text):
        t = re.sub('\n','。',text)
        ex = re.sub('\s','，',t)
        t = re.sub('\t','。',ex)

        return t

    def __csv(self,dt,file_path="csvfile.csv")->'save csv file':
        import csv
        _shape = np.shape(dt)

        with open(file_path,'w') as w:
            #delimiter可以设置每列值之间使用的隔开符，默认是每个值占一格。
            writer = csv.writer(w)
            if len(_shape)==1:
                writer.writerow(['id','name','val'])#写入一行。
            elif len(_shape)==2:
                #writerows()同时写入多行
                writer.writerows(dt)
            else:
                raise ValueError('too many dimension')


    def _docx(self)->"read .doc file":
        from docx import Document
        from docx.shared import Inches

        document = Document(self._file)

        _ds = []
        for p in document.paragraphs:
            if not p:
                continue
            else:
                _ds.append(self._re_row(p.text))
        
        self._read_result = _ds
    
    def _pdf(self)->"read .pdf file":
        import pdfplumber
        with pdfplumber.open(self._file) as pdf:
            # 逐行读取。
            _ps = []
            for c in pdf.pages:
                q = self._re_row(c.extract_text())
                _ps.append(q)
        
        self._read_result = _ps
    
    def _csv(self):
        import csv
        #读取
        c = open(self._file,'r')
        reader = csv.reader(c)
        _cs = []
        for f in reader:
            if f == '\x00':
                continue
            elif len(f)==0:
                continue
            else:
                s = '。'.join(f)
                _cs.append(self._re_row(s))
 
        self._read_result = _cs
    
    def _txt(self):
        with open(self._file,'r') as f:
            _ts = []
            for x in f:
                b = self._re_row(x)
                _ts.append(b)
        
        self._read_result = _ts

            



#replace_file(fp=r'/home/wcs/ITEM/selfTool',tp=r'/home/wcs/software/Anaconda/envs/wcs/lib/site-packages/selfTool')



