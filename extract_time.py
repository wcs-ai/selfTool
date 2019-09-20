#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
from datetime import datetime

h_date = datetime.now()
year = h_date.year
month = h_date.month
day = h_date.day

#春夏秋冬
#分级：年/5,月/4，日/3，时/2，分/1，秒/0
time_words = ['今年','明年','去年','前年','昨天','今天','前天','明天','后天','早上','中午','晚上','傍晚','今早']
reg = {
    'y1':re.compile('[今明去前]年'),
    'y2':re.compile('[0-9零一二两三四五六七八九]+年'),
    'y3':re.compile('年'),
    'm':re.compile('[^0][0-9]?月'),
    'm2':re.compile('月'),
    'd1':re.compile('[昨今前明后](天|日)'),
    'd2':re.compile('[1-3]?[0-9]?[零一二两三四五六七八九十]?(日|号)'),
    'd3':re.compile('日'),
    'h1':re.compile('早上|中午|晚上|傍晚|今早'),
    'h2':re.compile('(\d?\d|[一二两三四五六七八九十]+)(时|点|点钟)'),
    'mint':re.compile("(\d?\d|[一二两三四五六七八九十]+)分")
}

def comment_rank(word):
    grade = None
    ys1 = re.search(reg['y1'], word)
    ys2 = re.search(reg['y2'], word)
    ms = re.search(reg['m'], word)
    ds1 = re.search(reg['d1'], word)
    ds2 = re.search(reg['d2'], word)
    hs1 = re.search(reg['h1'], word)
    hs2 = re.search(reg['h2'], word)

    if ys1!=None or ys2!=None:
        grade = 5
    elif ms!=None:
        grade = 4
    elif ds1!=None or ds2!=None:
        grade = 3
    elif hs1!=None or hs2!=None:
        grade = 2

    return grade

#extract time.the arguments:(word,nature of word)
def extract(data):
    #assert len(data)==2,"data's length must be 2"
    #存储每个时间的数组
    words = []
    t_arr = []
    sentence = ''

    #保留前一个检测到的时间等级，与当前检测到的时间等级若或者大于等于前者新开一个数组值，否则放到前一个组中。
    pre_ga = 6
    #判断是否可添加
    add_sta = False
    for word,nature in data:
        #词性判断
        if nature=='m' or nature== 't':
            sentence += word
        else:
            if sentence!='':
                sentence += word
                words.append(sentence)
            sentence = ''
            continue

    for s in words:
        ts = word_to_time(s)
        t_arr.append(ts)


    return t_arr

def tran_num(sen):
    #点，点钟、分这几个字要去掉
    string = sen
    china_num = ['零','一','二','三','四','五','六','七','八','九','十','时','点','点钟','分']
    a_num = ['0','1','2','3','4','5','6','7','8','9','','','','','']
    for i in range(len(china_num)):
        if china_num[i]!='十':
            string.replace(china_num[i],a_num[i])
        else:
            #十的替换比较特殊
            ord = string.find('十')
            if ord==-1:
                continue
    
            if string[ord+1] in china_num and string[ord-1] in china_num:
                string.replace('十','')
            elif string[ord-1] not in china_num and string[ord+1] in china_num:
                string.replace('十','1')
            else:
                string.replace('十','0')

    return string

def word_to_time(sentence):
    global year,month,day
    r_y = ''
    r_m = ''
    r_d = ''
    r_h = ''
    r_min = ''
    #tm1放日期，tm2放具体时间，如果未匹配到具体时间则只返回tm1
    tm1 = []
    tm2 = []

    ys1 = re.search(reg['y1'], sentence)
    ys2 = re.search(reg['y2'], sentence)
    ms = re.search(reg['m'], sentence)
    ds1 = re.search(reg['d1'], sentence)
    ds2 = re.search(reg['d2'], sentence)
    hs2 = re.search(reg['h2'], sentence)
    min1 = re.search(reg['mint'],sentence)

#备注：未对个数年月日处理，如19年，55年等缩写简写。把检测到的带年，月，日等字去掉
    if ys1!=None:
        if(ys1.group()=='前年'):
            r_y = year - 2
        elif ys1.group()=='去年':
            r_y = year - 1
        elif ys1.group()=='明年':
            r_y = year + 1
    elif ys2!=None:
        r_y = tran_num(ys2.group())
    else:
        r_y = year

    #月份检测
    if ms!=None:
        r_m = tran_num(ms.group())
    else:
        r_m = month

    #check day
    if ds1!=None:
        if ds1.group() == '前天':
            r_d = day - 2
        elif ds1.group() == '昨天':
            r_d = day - 1
        elif ds1.group() == '明天':
            r_d = day + 1
        else:
            r_d = day
    elif ds2!=None:
        r_d = tran_num(ds2.group())
    else:
        r_d = day
    tm1.extend([r_y,r_m,r_d])

    #check hour
    if hs2!=None:
        r_h = tran_num(hs2.group())
    else:
        r_h = ''

    if min1!=None:
        r_min = tran_num(min1.group())
    else:
        r_min = '00'    
    tm2.extend([r_h,r_min])
    
    for i in range(len(tm1)):
        tm1[i] = str(tm1[i])

    if r_h=='':
        tm2_string = ''
    else:
        tm2_string = str(tm2[0])+":"+str(tm2[1])       

    return '-'.join(tm1)+","+tm2_string


