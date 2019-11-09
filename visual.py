#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
这个文件用来绘图
"""
import numpy as np
import matplotlib.pyplot as plt

def bar(x,y):
    plt.title('data bar')
    plt.figure(figsize=(50,80))
    plt.bar(x,y)
    plt.show()