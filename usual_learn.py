#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.utils import _joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from selfTool import _file
import decimal
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import pandas as pd

"""
class DecimalEncoder(json.JSONEncoder):
	def default(self,o):
		if isinstance(o,decimal.Decimal):
			for i,j in enumerate(o):
				o[i] = list(j)
			return list(o)
		super(DecimalEncoder,self).default(o)
"""


class bayes(object):
    def __init__(self, data, target, algorithm="GNB"):
        self.algorithm = algorithm
        self.data = data
        self.target = target
        if algorithm == 'GNB':
            self.model = GaussianNB()
        elif algorithm == 'MNB':
            self.model = MultinomialNB()
        elif algorithm == 'BNB':
            self.model = BernoulliNB()
        else:
            self.model = ComplementNB()

        self.model.fit(data, target)

    def save_model(self, path):
        _joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = _joblib.load(path)

    def predict(self, x):
        res = self.model.predict(x)
        return res


# 层次聚类树,[9,10,5]
class Layer_kmeans(object):
    def __init__(self, cluster=[]):
        self.MODEL = "Layer_kmeans"
        self._cluster = cluster
        self._clust_len = 0
        self._cluster_tree = {
            "position": 'root',
            "festival": [],
            "center_point": None
        }

    @property
    def result(self):
        return self._cluster_tree

    # arguments:the target data(mast be 2d),words with data,先分为9个类存为文件
    def tencent(self, data, words, clusters=[5]):
        _kmeans_tree = {
            "position": "root",
            "center_point": [],
            "festival": {}
        }

        class_data = {}

        one = clusters.pop(0)
        km = KMeans(init="k-means++", n_clusters=one)
        km.fit_predict(data)
        points = []

        for j, i in enumerate(km.cluster_centers_):
            key = 'file'+str(j)
            points.append(list(i))
            class_data[key] = {}
        _kmeans_tree['center_point'] = points

        # 将所有数据按类分开,存成字典
        for a, b in enumerate(km.labels_):
            key2 = 'file' + str(b)
            class_data[key2][words[a]] = data[a]

        # 各类存到不同的文件
        for idx in range(one):
            key1 = 'file' + str(idx)
            save_path = 'data/tree' + str(idx) + '.json'
            _kmeans_tree['festival'][key1] = save_path
            _file.op_file(file_path=save_path,
                         data=class_data[key1], model='json', method='save')
            # 保存后删除
            del class_data[key1]
        # 存储根节点查找文件
        _file.op_file(file_path='data/root.json',
                     data=_kmeans_tree, model='json', method='save')

    # 处理腾讯的9个词向量文件
    def take9_file(self, root_path):
        file_tree = {
            "tree0": 0,
            "tree1": 0,
            "tree2": 0,
            "tree3": 0,
            "tree4": 0,
            "tree5": 0,
            "tree6": 0,
            "tree7": 0,
            "tree8": 0
        }
        for f in range(3, 9):
            key = 'tree' + str(f)

            f_p = 'data/tencent/tree' + str(f) + '.json'
            file_tree[key] = _file.op_file(f_p, method='read')

            vals = list(file_tree[key].values())
            ks = list(file_tree[key].keys())

            sp = 'data/tencent/tc_tree' + str(f) + '.json'
            self.cluster(vals, ks, sp)
            del file_tree[key]
            del self._cluster_tree

            self._cluster_tree = {
                "position": 'root',
                            "festival": [],
                            "center_point": None
            }
            print(ord)

    # 这里开以开多线程操作,info with data(如果内存够用的话)
    def cluster(self, data, keys, save_path):
        self._clust_len = len(self._cluster) - 1
        self._basic_cluster(data, keys, self._cluster_tree, 0)

        _file.op_file(file_path=save_path, data=self._cluster_tree,
                     model='json', method='save')
        """
		***存储的数据结构：
		*{
		*	center_point:[],
		*	position:0,
		*	festival:[{
		*		center_point:[],
		*		position:last,
		*		festival:[{word1:val,word2:val,...}]
		*	},{...},...]
		*}
		"""

    # 参数：聚类数据、类数，当前层位置
    def _basic_cluster(self, data, keys, tree_obj, position=0):

        if position == 'last':
            n_clusters = self._cluster[self._clust_len]
        else:
            n_clusters = self._cluster[position]

        dts = []
        for v in range(n_clusters):
            dts.append({})

        # 当样本data长度小于n_clusters时
        n_clusters = len(data) if len(data) < n_clusters else n_clusters

        km = KMeans(init="k-means++", n_clusters=n_clusters)
        km.fit_predict(data)

        # 将得到的各类别分开
        for i, j in enumerate(km.labels_):
            dts[j][keys[i]] = data[i]

        # 利用对象传参是按引用传递的方法来完善整颗树。得到的点
        center_data = []
        for cd in km.cluster_centers_:
            center_data.append(list(cd))

        tree_obj['center_point'] = center_data
        tree_obj['position'] = position

        if position != 'last':
            for i, g in enumerate(dts):
                tree_obj['festival'].append({
                    "center_point": 0,
                    "festival": [],
                    "position": ''
                })
                pt = 'last' if position + \
                    1 == self._clust_len else (position + 1)
                next_keys = list(g.keys())
                next_values = list(g.values())
                self._basic_cluster(next_values, next_keys,
                                    tree_obj['festival'][i], pt)
        else:
            # 至此一个循环完成
            tree_obj['festival'] = dts

    # 从腾讯词向量中查找相似
    def search_tencent(self, dts, root_path, branchs=2, candidate=1, distance=3):
        root = _file.op_file(root_path, method='read')
        dist = {}
        for idx, i in enumerate(root['center_point']):
            val = data.point_distance(dts, i)
            dist[round(val, 3)] = idx

        keys = list(dist.keys())
        keys.sort()
        sel_point = len(root['center_point']) if len(
            root['center_point']) < branchs[0] else branchs[0]
        branchs.pop(0)

        all_res = []
        for j in range(sel_point):
            k = 'file' + str(dist[keys[j]])
            path = root['festival'][k]
            sr = self.similirity(dts, path, branchs, candidate, distance)
            all_res.extend(sr)

        boult = [g for g in sorted(all_res, key=lambda k:k[0], reverse=False)]
        save_len = candidate if candidate < len(boult) else len(boult)
        return boult[0:save_len]

    # 查找相似数据，data,file,查找最近的两个分支，最多保留5个值,最大匹配距离，超过该值则剔除
    def similirity(self, data, file_path, branchs=[2, 2], candidate=3, distance=15):

        self._max_dist = distance
        self._search_branch = branchs
        self._search_result = []

        result = _file.op_file(file_path, model='json', method='read')
        self.search_tree(data, result)

        sr = [c for c in sorted(self._search_result,
                                key=lambda k:k[0], reverse=False)]
        save_len = candidate if candidate < len(sr) else len(sr)
        return sr[0:save_len]

    def search_tree(self, dts, tree):
        center_distance = {}
        # 与各质心点计算距离，排序，选择点个数。
        for idx, i in enumerate(tree['center_point']):
            dist = data.point_distance(dts, i)
            # 距离为键，索引为值
            center_distance[round(dist, 3)] = idx

        keys = list(center_distance.keys())
        keys.sort()

        pdx1 = tree['position'] if tree['position'] != 'last' else (
            len(self._search_branch)-1)
        sel_point = len(tree['center_point']) if len(
            tree['center_point']) < self._search_branch[pdx1] else self._search_branch[pdx1]

        index_arr = []
        for j in range(sel_point):
            if keys[j] > self._max_dist:
                break
            else:
                # 找到最小距离分支的索引
                index_arr.append(center_distance[keys[j]])
        # 不是最后一层则向下查找
        if tree['position'] != 'last':
            for m in index_arr:
                self.search_tree(dts, tree['festival'][m])
        else:
            last_festival = []

            # 至此完成一个循环
            for n in range(sel_point):
                # 每条数据是:距离、对应的数据信息
                last_festival.append(tree['festival'][index_arr[n]])
            # 将最近距离的几个节点放到last_festival中
            for t in last_festival:
                dist_obj = {}
                for v in t:
                    # 计算每个节点中与目标的距离
                    dist_obj[v] = data.point_distance(dts, t[v])
                # words key array
                sort_dist = [y[0] for y in sorted(
                    dist_obj.items(), key=lambda s:s[1], reverse=False)]

                pdx2 = tree['position'] if tree['position'] != 'last' else (
                    len(self._search_branch)-1)

                sel_len = len(sort_dist) if len(
                    sort_dist) < self._search_branch[pdx2] else self._search_branch[pdx2]
                # 保留距离最近的几个
                sel_res = sort_dist[0:sel_len]

                for g in sel_res:
                    self._search_result.append([dist_obj[g], {g: t[g]}])


class RelationCalc(object):

    def _pearson(self, x, y) -> '皮尔逊相关系数':
        _a = pearsonr(x, y)
        return _a

    def _spearman(self, x, y) -> '斯皮尔曼系数':
        _p = spearmanr(x, y, axis=0, nan_policy='omit')
        return _p

    def _kendal(self, x, y) -> '肯德尔系数':
        _k = kendalltau(x, y, nan_policy='omit')
        return _k

    def _cov(self, x, y):
        _scalx = np.max(x) - np.mean(x)
        _scaly = np.max(y) - np.mean(y)
        # 映射到-1～1
        return np.cov(x, y)[0][1] / (_scalx * _scaly)

    def _mutualInfo(self, x, y) -> '互信息计算':
        _counter_x = dict()
        _counter_y = dict()
        _counter_xy = dict()

        assert len(x) == len(y), 'x unequal y'

        NUMS = len(x)
        # 统计各项值出现的频率

        def _counter(key, obj):
            if key in obj:
                obj[key] += 1
            else:
                obj[key] = 1

        for a, b in zip(x, y):
            _key1 = str(a)
            _key2 = str(b)

            _key1_and_2 = _key1 + '-' + _key2

            _counter(_key1, _counter_x)
            _counter(_key1_and_2, _counter_xy)
            _counter(_key2, _counter_y)

        XY_NUMS = 0
        for i in _counter_xy:
            XY_NUMS += _counter_xy[i]
        # 计算互信息值
        _res = 0
        for v in _counter_xy:
            ks = v.split('-')

            _pxy = _counter_xy[v] / XY_NUMS
            _px = _counter_x[ks[0]] / NUMS
            _py = _counter_y[ks[1]] / NUMS

            _res += _pxy * np.log2(_pxy / (_px * _py))

        return _res

    def calc(self, d1, d2, fn_str='pearson'):
        if fn_str == 'pearson':
            q = self._pearson(d1, d2)[0]
        elif fn_str == 'spearman':
            q = self._spearman(d1, d2)[0]
        elif fn_str == 'kendal':
            q = self._kendal(d1, d2)[0]
        elif fn_str == 'cov':
            q = self._cov(d1, d2)
        elif fn_str == 'mutualInfo':
            q = self._mutualInfo(d1, d2)
        else:
            raise ValueError('dont support {}'.format(fn_str))

        return q


class Apriori(object):
    _DISCRIBE = "Apriori algorithm"
    def __init__(self, datas):
        self._data = datas
        self._NUM = len(datas)
    # 最小置信度、单项最小支持度、组的最小支持度
        self._confidence = 0.7
        self._minValSupport = 3
        self._minGroupSupport = 2
        self._aloneSupports = dict()
        self._itemSupports = None

    def _calcAloneSupport(self):
        for a in self._data:
            for b in a:
                if b not in self._aloneSupports:
                    self._aloneSupports[b] = 1
                else:
                    self._aloneSupports[b] += 1

    def _moveAlone(self):
        # 移除
        _move_keys = [c for c in self._aloneSupports if self._aloneSupports[c] < self._minValSupport]

        _copy_val = self._aloneSupports.copy()
        for i in _move_keys:
            _copy_val.pop(i)
        #_range = [i[0] for i in sorted(_copy_val.items(),key=lambda k:k[1],reverse=True)]
        return _copy_val

    def group(self,c=(2,3)):
        # c分组的个数
        import itertools
        from selfTool import base
        _moves = self._moveAlone()
        # 所有分组情况,{"a0":{'support':1,'val':[1,2],'max':3,'confidence':0.8}}
        _groups = dict()
        
        _idx = 0
        
        def _count(ds):
            nonlocal _idx

            for i in ds:
                _listi = list(i)
                _key = 'a' + str(_idx)
                _idx += 1
                for j in self._data:
                    # i是否在j中,计算各项支持度
                    if base.objInclude(_listi,j)==True:
                        if _key in _groups:
                            _groups[_key]['support'] += 1
                        else:
                            # 计算该项前件
                            _vals = [_moves[a] for a in i]
                            _max = max(_vals)
                            _max_idx = _vals.index(_max)
                            _groups[_key] = {"val":i,"support":1,"max":i[_max_idx],'maxVal':_max,'confidence':0}
                    else:
                        continue
                                                 
        for j in c:
            q = itertools.combinations(list(_moves),j)
            _count(q)
        
        self._itemSupports = _groups
    
    def implement(self):
        self._calcAloneSupport()
        self.group()
    
    def run(self,confidence=0.7):
        # 计算置信度并剔除小的。
        new_obj = dict()
        for k in self._itemSupports:
            _confidence_val = self._itemSupports[k]['support'] / self._itemSupports[k]['maxVal']
            self._itemSupports[k]['confidence'] = _confidence_val
            if _confidence_val >= confidence:
                new_obj[k] = self._itemSupports[k]
            else:
                continue
        
        return new_obj
        


class TimeOrd(object):
    """
    自制：
    描述不明确的波动数据，计算其下一期可能增大或减小的可能性。-表示减，正值表示增。
    """
    def __init__(self,dts):
        # dts:[1,6,3,...]
        assert len(dts)>0,print(dts)
        self._seqson = []
        self._dts = dts
        self._mean = np.mean(dts)
        self._max = np.max(dts)
        self._min = np.min(dts)
        # 极大值均值，极小值均值、平均差值、极值点索引位置。
        self._max_mean = 0
        self._min_mean = 0
        self._var = 0
        self._max_min_idx = []
        
        # _searchEqual()中的缓冲值。
        self._buffleVal = ''
        
        self._superProperty()
    
    def _searchEqual(self,val,start,dis=1):
        # 递归查找与val平等相邻但不同的值与其结果。
        #  递归的程序中，return返回的值其变量拿不到，特此用全局值
        if start<0 or start > len(self._dts)-1:
            self._buffleVal = 0
        elif self._dts[start]==val:
            self._searchEqual(val,start+dis,dis)
        else:
            self._buffleVal = self._dts[start] - val
    
    def _superProperty(self):
        _maxs = []
        _mins = []
        _var = 0
        
        for i in range(1,len(self._dts)-1):
            self._searchEqual(self._dts[i],i,-1)
            _lf = self._buffleVal
            self._searchEqual(self._dts[i],i,1)
            _rg = self._buffleVal
            
            _var = abs(self._dts[i-1] - self._mean)
            
            if _lf >= 0 and _rg >= 0:
                _maxs.append(self._dts[i])
                self._max_min_idx.append(i)
            elif _lf <= 0 and _rg <= 0:
                _mins.append(self._dts[i])
                self._max_min_idx.append(i)
        
        _var += abs(self._dts[-1] - self._mean)
        self._var = _var / len(self._dts)
        
        # 添加极大、极小平均值
        if len(_maxs)==0 and len(_mins)!=0:
            self._max_mean = self._min_mean = np.mean(_mins)
        elif len(_maxs)!=0 and len(_mins)==0:
            self._max_mean = self._min_mean = np.mean(_maxs)
        else:
            self._max_mean = np.mean(_maxs)
            self._min_mean = np.mean(_mins)
    
    def norm(self):
        """不规则的时间序列。根据前几期的值，计算下一期减小或增大的可能性。
        为负表示继续减小的可能性，为正表示继续增大的可能性，其度量用它们的绝对值表示。
        暂时没有考虑值之间的出现间隔！！
        
        可以用：val + val * trend得到的值与目标属性做一个皮尔逊系数检验，一般能得到不错的相关性。   o
        """
        trends = []
        # 计算前3加权斜率。
        def _cSlope(xs):
            g = len(xs)
            assert g<=4,'many el'
            ws = [0.1,0.3,0.6] if g==4 else [0.3,0.6]
            
            _m = 0
            _ts = []
            # 为0的话，range是一个空对象。
            for v in range(g-1):
                q = (xs[v+1] - xs[v]) / 1
                _ts.append(q * ws[v])
            return sum(_ts)

        # 计算控制系数。
        def _control(v):
            if v >= self._max or v <= self._min:
                return -0.1
            else:
                return 1
        
        _slopes = 0
        _max_min = self._max - self._min
        # 计算每一个点的变化趋势。
        for i,t in enumerate(self._dts):
            if i==0:
                slope = 0
            else:
                _stx = 0 if i <= 3 else i - 3
                _useVal = self._dts[_stx:i+1]
                assert len(_useVal)>1,'{} to {}'.format(_stx,i+1)
                slope = _cSlope(_useVal)

            if slope==0:
                trends.append(0)
                continue
            
            _slopes += slope
            
            positive_negative = slope / abs(slope)
            c = _control(t)
            
            coefficient = ((1 + positive_negative) * self._max / 2) - (positive_negative * t) - ((1 - positive_negative) * self._min / 2)
            
            p = c * (slope + positive_negative * coefficient) / _max_min
            
            """
            _dist = (t - self._mean) / (self._var + 1)
            if (slope < 0 and _dist >= 0) or (slope > 0 and _dist <= 0):
                trend = -_dist + slope
            elif slope < 0 and _dist <= 0:
                trend = slope - 1 + abs(_dist)
            elif slope > 0 and _dist >= 0:
                trend = slope + 1 - _dist
            else:
                trend = 0
            """
            trends.append(round(p,4))
        
        # 趋势为0的用总平均趋势代替。
        _mean_slope = _slopes / len(self._dts)
        for i,j in enumerate(trends):
            if j==0:
                trends[i] = _mean_slope
            else:
                continue
        
        return trends

    def timeDist(self,dts,window=5,qsort=False):
        DLEN = len(dts)
        assert DLEN>0,'empty data'

        SMOOTH = 1.2
        
        if qsort:
            dts = np.sort(dts)
        
        res_array = []
        
        for i in range(DLEN):
            ix = 0 if i<window else i - window
            # 前几次记录。
            use_dt = dts[ix:i]
            ulen = len(use_dt)
            
            if ulen==0:
                val = 0.01
            elif ulen==1:
                val = use_dt[0] / (max([dts[i] - use_dt[0],use_dt[0]]) * SMOOTH)
            else:
                dist_array = []
                # 计算每两个相邻时间的差值。
                for j in range(1,ulen):
                    dist_array.append(use_dt[j] - use_dt[j-1])
                
                _mean = np.mean(dist_array)
                # 距离上一次时间 / 平均记录时间。
                val = (dts[i] - use_dt[-1] + 0.95) / (_mean + 1)
                
            res_array.append(val)
            
        return res_array

        
        
        