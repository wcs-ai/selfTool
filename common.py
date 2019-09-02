#!/usr/bin/python
#-*-coding:UTF-8-*-
import threading
import queue,os
threads = []
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



