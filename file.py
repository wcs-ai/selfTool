#!/usr/bin/python
#-*-coding:UTF-8-*-
import os

#读取英文文本,参数：文件路径列表
def read_en_txt(sequence,coding):
    assert type(sequence)!='list',"inputs is't a list"
    coding = coding or 'gb2312'
    words_list = []
    words_obj = {}

    en_stop = ['__',':',',','.','?','-','!']
    for file in sequence:
        assert os.path.isfile(file)==True,"fileError"
        with open(file,'r',encoding=coding) as f:
            f1 = f.read().replace('\n','')
            f2 = f1.lower().replace('<br />','').split()
            for word in f2:
                if word in en_stop:
                    continue
                else:
                    if word in words_list:
                        pass
                    else:
                        words_list.append(word)

                    if word in words_obj:
                        words_obj[word] += 1
                    else:
                        words_obj[word] = 1

    return words_list,words_obj


