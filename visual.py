#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
这个文件用来绘图
"""
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go
import random
from plotly import tools
import plotly.offline as ply
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('WX')


class Draw(object):
    def __init__(self):
        self._unify_config = {
            "title": "图形",
            "figure": [5, 3],
            "font": "SimHei",
            "xlabel": "x轴标题",
            "ylabel": "y轴标题",
        }

        self._bar_config = {
            "width": 0.5,  # 每个条形宽度
            "color": "red"
        }

        self.unify_draw()

    def unify_draw(self):
        # 一些统一的绘图。
        matplotlib.rcParams['font.sans-serif'] = [self._unify_config["font"]]
        plt.title(self._unify_config["title"])
        plt.figure(figsize=self._unify_config["figure"])

    def bar(self, data=None):
        """绘制柱状图(条形图)"""
        dt = data if data != None else [15, 50, 22, 80]
        _indexs = range(len(dt))
        print(dt)
        plt.bar(
            _indexs, dt, color=self._bar_config["color"], width=self._bar_config["width"])
        plt.show()


class PlotlyDraw(object):
    _DICCRIBE = "plotly绘图"

    def __init__(self):
        """
        array = [{
            "x":[],
            "y":[]
        }]
        """
        self._colors = ['red','blue','yellow','green','orange','violet']
        # 这里写所有配置，更改时请用update()方法。
        self._config = {
            "subplot": [1, 1],
            "layout": {
                "title": '数据分析图',
                "plot_bgcolor": '#E6E6FA',  # 图的背景颜色
                "paper_bgcolor": '#F8F8FF',  # 图像的背景颜色
                "autosize": True,
                "yaxis2": dict(title="大数值", overlaying='y', side="right")
            },
            # layout中axis,yaxis属性示例。
            "axis": {
                "autorange": True,
                # range:(0, 100),
                "dtick": 10,
                "showline": True,
                "mirror": 'ticks'
            },
            "save_path": 'data_visual.html'
        }

    @property
    def config(self):
        return self._config

    def scatter(self, x, y, mode='markers', name="散点图", marker={}, line={}):
        _marker = {
            "size": 3,
            "color": 'rgb(49,130,189)',
            "opacity": 1,
            "showscale": True
        }
        _line = {
            "width": 2,
            "color": 'blue'
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

    def bar(self, x, y, name="条形图", marker={}):
        _marker = {
            "color": 'rgb(49,130,189)',
            "width": 10
        }
        _marker.update(marker)
        return go.Bar(x=x,
                      y=y,
                      name=name,
                      marker=_marker)

    def distributionAnalysisGraph(self, dt, feature, classes,scalar=1,filename='distribute.html'):
        """用于绘制分布分析图：只针对feature[1]是分类型情况。
        dt:pd.DataFrame();
        feature:[x,y],要分析的特征,对应的类别特征项;
        classes:分类型，模型对应的分类数。[0,1,2];
        """
        _ys = []
        _lens = []
        datas = []
        # 按各类别对特征项分组
        for c in classes:
            _val = dt[feature[0]][dt[feature[1]] == c]
            if len(_val) < 200:
                _ys.append(_val * scalar)
                _lens.append(len(_val))
            else:
                _ys.append(_val[0:200] * scalar)
                _lens.append(len(_val))
        _xs = list(range(max(_lens)))

        i = 0
        for y,k in zip(_ys,classes):
            trace = go.Scatter(
                x=_xs,
                y=y,
                mode="markers",
                name="类{}".format(k),
                marker=dict(
                    size=10,
                    color=self._colors[i],
                    showscale=True
                )
            )
            datas.append(trace)
            i += 1
        
        ply.plot(datas,filename=filename)
            

    def batch_draw(self, datas, cls=1):
        """批量多子图绘制
        datas:[{"x":[],"y":[],"graph_type":'',"graph":go.scatter(),"args":{}}]
        """
        import math
        assert cls < 3, 'too many column'
        self._config['subplot'] = [len(datas), 1] if cls == 1 else [
            round(len(datas) / 2), 2]
        _subplot = tools.make_subplots(
            self._config['subplot'][0], self._config['subplot'][1], print_grid=False)
        _fig = go.FigureWidget(_subplot)

        for ix, d in enumerate(datas):
            # 没有传入图列的情况。
            if 'graph' not in d:
                #assert type(d['args'])==dict,'query argument:args'
                if d['graph_type'] == 'scatter':
                    d['graph'] = self.scatter(
                        x=d['x'], y=d['y'], name=d['name'])
                elif d['graph_type'] == 'bar':
                    d['graph'] = self.bar(x=d['x'], y=d['y'], name=d['name'])

            self._config['layout']['xaxis{}'.format(
                ix + 1)] = self._config['axis']
            self._config['layout']['yaxis{}'.format(
                ix + 1)] = self._config['axis']

            _use_row = ix + 1
            if cls == 1:
                _row = _use_row
                _col = 1
            else:
                _row = math.ceil(_use_row / 2)
                _col = 2 if _use_row % 2 == 0 else 1

            _fig.add_trace(d['graph'], row=_row, col=_col)

        _fig['layout'].update(self._config['layout'])

        ply.plot(_fig, filename=self._config['save_path'])
