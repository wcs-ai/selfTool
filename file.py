#!/usr/bin/python
#-*-coding:UTF-8-*-
import os,queue

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

#replace_file(fp=r'E:\AI\selfTool',tp=r'D:\develope\Anaconda\envs\wcs\Lib\site-packages\selfTool')



