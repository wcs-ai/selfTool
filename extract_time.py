#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re
import datetime

time_words = ['今年','明年','去年','前年','昨天','今天','前天','明天','后天','早上','中午','晚上','傍晚','今早']
reg = {
    'year':re.compile('[今明去前]年'),
    'day':re.compile('[昨今前明后]天'),
    'h':re.compile('早上|中午|晚上|傍晚|今早')
}

def extract(data):
    assert len(data)==2,"data's length must be 2"
