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


import plotly.offline as py
from plotly import tools
import random
import plotly.graph_objs as go
from plotly.graph_objs import Scatter,Layout



class PlotlyDraw(object):
    _DICCRIBE = "plotly绘图"
    def __init__(self):
        """
        array = [{
            "x":[],
            "y":[]
        }]
        """
        # 这里写所有配置，更改时请用update()方法。
        self._config = {
            "subplot":[1,1],
            "layout":{
                "title":'数据分析图',
	            "plot_bgcolor":'#E6E6FA',#图的背景颜色
                "paper_bgcolor":'#F8F8FF',#图像的背景颜色
                "autosize":True,
	            "yaxis2":dict(title="大数值", overlaying='y', side="right")
            },
            # layout中axis,yaxis属性示例。
            "axis":{
                "autorange":True,
                #range:(0, 100),
                "dtick":10,
                "showline":True,
                "mirror":'ticks'
            },
            "save_path":'data_visual.html'
        }
    
    @property
    def config(self):
        return self._config

    def scatter(self,x,y,mode='markers',name="散点图",marker={},line={}):
        _marker = {
            "size":3,
            "color":'rgb(49,130,189)',
            "opacity":1,
            "colorscale":'Viridis',
            "showscale":True
        }
        _line = {
            "width":2,
            "color":'blue'
        }

        _line.update(line)
        _marker.update(marker)

        return go.Scatter(x=x,
                        y=y,
                        mode=mode,
                        name=name,
                        text='p',
                        textposition='bottom center',
                        textfont={'size': 12},  
                        marker=_marker,
                        line=_line)

    def bar(self,x,y,name="条形图",marker={}):
        _marker = {
            "color":'rgb(49,130,189)',
            "width":10
        }
        _marker.update(marker)
        return go.Bar(x=x,
                      y=y,
                      name=name,
                      marker=_marker)

    def batch_draw(self,datas,cls=1):
        """批量多子图绘制
        datas:[{"x":[],"y":[],"graph_type":'',"graph":go.scatter(),"args":{}}]
        """
        import math
        assert cls < 3,'too many column'
        self._config['subplot'] = [len(datas),1] if cls==1 else [round(len(datas) / 2),2]
        _subplot = tools.make_subplots(self._config['subplot'][0], self._config['subplot'][1], print_grid=False)
        _fig = go.FigureWidget(_subplot)

        for ix,d in enumerate(datas):
            # 没有传入图列的情况。
            if 'graph' not in d:
                #assert type(d['args'])==dict,'query argument:args'
                if d['graph_type']=='scatter':
                    
                    d['graph'] = self.scatter(x=d['x'],y=d['y'])
                elif d['graph_type']=='bar':
                    d['graph'] = self.bar(x=d['x'],y=d['y'])

            self._config['layout']['xaxis{}'.format(ix + 1)] = self._config['axis']
            self._config['layout']['yaxis{}'.format(ix + 1)] = self._config['axis']

            _use_row = ix + 1
            if cls==1:
                _row = _use_row
                _col = 1
            else:
                _row = math.ceil(_use_row / 2)
                _col = 2 if _use_row % 2==0 else 1

            _fig.add_trace(d['graph'],row=_row,col=_col)

        _fig['layout'].update(self._config['layout'])

        py.plot(_fig,filename=self._config['save_path'])

