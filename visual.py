#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
这个文件用来绘图
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('WX')
import matplotlib.pyplot as plt


class Draw(object):
	def __init__(self):
		self._unify_config = {
			"title":"图形",
			"figure":[5,3],
			"font":"SimHei",
			"xlabel":"x轴标题",
			"ylabel":"y轴标题",
		}

		self._bar_config = {
			"width":0.5, #每个条形宽度
			"color":"red"
		}

		self.unify_draw()

	def unify_draw(self):
		# 一些统一的绘图。
		matplotlib.rcParams['font.sans-serif'] = [self._unify_config["font"]]
		plt.title(self._unify_config["title"])
		plt.figure(figsize=self._unify_config["figure"])


	def bar(self,data=None):
		"""绘制柱状图(条形图)"""
		dt = data if data!=None else [15,50,22,80]
		_indexs = range(len(dt))
		print(dt)
		plt.bar(_indexs,dt,color=self._bar_config["color"],width=self._bar_config["width"])
		plt.show()


"""
import plotly_express as px	
gapminder = px.data.gapminder()	
gapminder2007 = gapminder.query('year == 2007')	
px.scatter(gapminder2007, x='gdpPercap', y='lifeExp')

import plotly_express as px
gapminder = px.data.gapminder()
gapminder2007 = gapminder.query('year == 2007')
fig = px.scatter(gapminder2007, x='gdpPercap', y='lifeExp')
fig.show()
"""
from plotly.graph_objs import *
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go
from plotly.graph_objs import Scatter,Layout

#Create traces
trace0 = go.Bar(
    x = ['Jan','Feb','Mar','Apr', 'May','Jun',
         'Jul','Aug','Sep','Oct','Nov','Dec'],
    y = [20,14,25,16,18,22,19,15,12,16,14,17],
    name = 'Primary Product',
    marker=dict(
        color = 'rgb(49,130,189)'
    )
)
trace1 = go.Bar(
    x = ['Jan','Feb','Mar','Apr', 'May','Jun',
         'Jul','Aug','Sep','Oct','Nov','Dec'],
    y = [19,14,22,14,16,19,15,14,10,12,12,16],
    name = 'Secondary Product',
    marker=dict(
        color = 'rgb(204,204,204)'
    )
)
labels = ['产品1','产品2','产品3','产品4','产品5']
values = [38.7,15.33,19.9,8.6,17.47]
#    饼图与上面稍有不同
trace = [go.Pie(labels=labels, 
				values=values,
				#hole=0.7,	#绘制成同心圆
				hoverinfo = 'label+percent', 
				pull = [0.1,0,0,0,0],#设置各部分的突出程度。
                textinfo = 'percent', # textinfo = 'value',
                textfont = dict(size = 30, color = 'white'),
				#	设置各元素属性。line设置边框属性。
				marker=dict(colors=['#FFFF00', '#FF0000', '#E066FF', '#0D0D0D'],
							line=dict(color = '#000000', width = 2)
							)
				)]
layout = go.Layout(
    title = '产品比例配比图',
)
fig = go.Figure(data = trace, layout = layout)
data = [trace0,trace1]

py.plot(fig,filename="resource/bing.html")
#py.plot(trace,filename="resource/bing.html")