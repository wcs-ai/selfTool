#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 
"""
这个文件用来做数据预处理
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json,random
import math,re
import copy,tqdm
import jieba
import nltk
from selfTool import _file,common,usual_learn
from scipy import stats

#数据预处理模块
class Dispose(object):
    def __init__(self):
        self.MODEL = 'data-dispose'
        self.data_type = [list,tuple,np.ndarray,pd.core.frame.DataFrame]
        self.dim = 2
        self.miss_val = [None]
        self.place = 0
    
    # 设置miss值。
    @property
    def miss(self):
        return self.miss_val
    @miss.setter
    def miss(self,val):
        self.miss_val = val

    #查找所有缺失值位置
    def search_miss(self,data):
        miss_index_arr = []
        #允许的数据类型
        for v1,val1 in enumerate(data):
            if type(val1) in self.data_type:
                self.dim = 2
                for v2,val2 in enumerate(val1):
                  #val2是numpy等类型的情况需要变为list
                    try:
                        va2 = list(val2)
                    except:
                        va2 = val2
                    if va2 in self.miss_val:
                        miss_index_arr.append([v1,v2])
            else:
                self.dim = 1
                try:
                    va1 = list(val1)
                except:
                    va1 = val1
                if va1 in self.miss_val:
                    miss_index_arr.append(v1)
        return miss_index_arr

    def _window_data(self,data,idx,window=5):
        """求出idx附近可用的值
        args:
        data:1d;要求是pd.DataFrame()数据。
        idx:select index;
        window:idx left。idx right。
        """
        if idx >= window:
            _ix1 = idx - window
        else:
            _ix1 = 0
            
        if len(data) <= (idx + window):
            _ix2 = len(data)
        else:
            _ix2 = idx + window
        
        _ixs = list(range(_ix1,idx)) + list(range(idx+1,_ix2))
        _val = data[_ixs][data.notnull()]
        
        return _val,_val.index

    def _mean(self,data):
        """用来计算指定数据的均值。
        """
        
        if len(data)==0:
            res = self.place
        else:
            res = np.mean(data, axis=0)
        return res

    def checkOutliersAndNone(self,x,method='box',discrete=None,scope=None):
        """检测异常值和None值。会连缺失值一起算在内。 
        x:[1,2,3,...];
        method:3a(3a原则),box(箱形图);
        discrete:method为非box和3a时使用。
        """
        _arr = pd.DataFrame({'data':list(x)})
        if method=='3a':
            _std = np.std(x)
            _outliers = _arr['data'][_arr['data']>3*_std].index
        
            # 一维特征情况可直接使用index当目标索引。
        elif method=='box':
            #箱形图，四分位点，外限点。
            _hig = np.quantile(x,0.75,interpolation='lower')
            _lower = np.quantile(x,0.25,interpolation='higher')
            _val = 3 * (_hig - _lower)
            _outliers = _arr['data'][(_arr['data']>(_hig + _val))|(_arr['data']<(_lower - _val))].index
        elif method=='scope':
            # 先验知识，指定数据范围
            _outliers = _arr['data'][(_arr['data']<scope[0])|(_arr['data']>scope[1])].index
        elif method=='discrete':
            # 对离散型数据做异常值检测
            _outliers = []
            assert discrete != None,'query discrete'
            #_outliers = _arr['data'][_arr['data'].isin(discrete)].index
            for ix,val in enumerate(x):
                if val not in discrete:
                    _outliers.append(ix)
        else:
            _outliers = []
        
        _none_val = _arr['data'][_arr['data'].isnull()].index
        # 返回异常值所在索引,和缺失值所在索引。       
        _all_set = set(list(_outliers))
        _non_set = set(list(_none_val))
        _out_set = _all_set - _non_set
        
        return (list(_out_set),list(_non_set))
    
    
    def batch_checkOutliersAndNone(self,dt,query_print=False,**args):
        """批量检测异常值
        data:pd.DataFrame();
        args:{
            "discrete":[{"name":'a',val:[1,2]},...],    //指定几种类型的检测
            "scope":[{"name":'a',val:[1.2,...]}],   //指定范围内的检测方法
            "box":[],
            "3a":[]
        }
        """
        _res_list = []
        for key in args:
            for d in args[key]:
                if key=='discrete':
                    _res = self.checkOutliersAndNone(x=dt[d['name']],method=key,discrete=d['val'])
                elif key=='scope':
                    _res = self.checkOutliersAndNone(x=dt[d['name']],method=key,scope=d['val'])
                else:
                    _res = self.checkOutliersAndNone(x=dt[d['name']],method=key)
                    
                _res_list.append({"name":d['name'],"outliers":_res[0],"nons":_res[1]})
                
                if query_print==True:
                    print('{}：异常值=>{}条;缺失值=>{}条;part value=>{}'.format(d['name'],len(_res[0]),len(_res[1]),dt[d['name']][_res[0][0:10]]))
        
        return _res_list
            
    
    def _alone_var_interpolate(self,x,method="lagrange"):
        """单变量插补,建议连续型数据使用。
        x:[1,2,3,4,5]#np.ndarry()
        """
        from scipy.interpolate import lagrange

        miss_idx = list(x[x.isnull()].index)

        for val in miss_idx:
            # 查找缺失值附近可用的值。
            _apply_data,idxs = self._window_data(x,val,window=50)
            #判断插值方法,mean:使用缺失值附近的数据的均值代替
            if method=='mean':
                res = self._mean(_apply_data)
            elif method=='lagrange':
                res = lagrange(idxs,_apply_data)(val)

            x[val] = res
            
        assert len(x[x.isnull()])==0,'exist None'
        return x
    
    def _multi_var_interpolate(self,dt,use_columns,target,method="cknn"):
        """多变量插补,要求其它维特征没有缺失值。
        x：可以是1d或2d。
        y：要插值的特征列。
        """
        from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
        
        assert isinstance(dt, pd.core.frame.DataFrame),'dt must be pd.core.frame.DataFrame'

        #选出拟合用的数据
        _fit_indexs = dt[target][dt[target].notnull()].index
        _none_indexs = dt[target][dt[target].isnull()].index
        
        # 对一维数据做了扩维处理
        if len(use_columns)==1:
            #__x = [[0,i] for i in ]
            _fitx = np.reshape(dt[use_columns[0]][_fit_indexs],(-1,1))
            #_predx = [[0,j] for j in ]
            _predx = np.reshape(dt[use_columns[0]][_none_indexs],(-1,1))
            assert len(_predx)>0,'no None'
        else:
            _fitx = dt.loc[_fit_indexs][use_columns].values
            _predx = dt.loc[_none_indexs][use_columns].values
            
        _fity = dt.loc[_fit_indexs][target].values
        assert len(_fitx)==len(_fity),'x no same len than y.'

        if method=='cknn':
            # 最近邻分类插值。
            _knn = KNeighborsClassifier(n_neighbors=3,#返回候选个数
                        algorithm='auto',
                        leaf_size=30,
                        weights='uniform')
        elif method=='rknn':
            _knn = KNeighborsRegressor(n_neighbors=2,algorithm='auto',weights='uniform')
            
        _knn.fit(_fitx,_fity)
        _pred = _knn.predict(_predx)
        
        dt[target][_none_indexs] = _pred

        # 返回目标列插值结果。
        return dt

        
    def interpolate(self,dt,use_columns=None,target='y',model='alone',method="mean"):
        """对给定缺失值类型进行插值.
        args:
        data:2d；model为alone时需要1d，multi时需要为pd.DataFrame()二维。
        target:选择多变量插补时需要，要插值的目标列
        method:插值方法.
        model:'alone'=>单变量插值。'multi'=>多变量插值。
        place:没有好的插值方案时使用的占位值.
        """
        import copy

        if model=='alone':
            if type(dt)==list or type(dt)==tuple or type(dt)==np.ndarray:
                dt = pd.DataFrame({'a':dt})

            _res = self._alone_var_interpolate(dt,method=method)
        else:
            _res = self._multi_var_interpolate(dt,use_columns=use_columns,target=target,method=method)
        
        return _res
    
    
    def norm_data(self,data,algorithm='norm'):
        #数据规范化,dt:2d
        dt = np.array(list(data))
        shape = np.shape(dt)
        _custom = True if algorithm[0:2]=='u-' else False 
        
        if _custom==False and len(shape)<=1:
            raise ValueError('query 2d data')


        if algorithm=='norm':
            scaler = preprocessing.normalize
        elif algorithm=='max-min':
            # 最大最小归一化
            scaler = preprocessing.MinMaxScaler()
        elif algorithm=='qt':
            scaler = preprocessing.QuantileTransformer()
        elif algorithm=='max':
            # 最大绝对值归一化
            scaler = preprocessing.MaxAbsScaler()
        elif algorithm=='stand':
            # 减均值，比方差
            scaler = preprocessing.StandardScaler()
        elif algorithm=='u-max':
            # 自定义，最大规范化，规范到-1~1之间，这里应该改为求dt中每个值的绝对值的最大值
            _max = abs(np.max(dt))
            _res = dt / _max
        elif algorithm=='u-max-min':
            # 自定义最大最小归一化，规范到0~1之间
            _max = np.max(dt)
            _min = np.min(dt)
            _res = (dt - _min) / (_max - _min)
        elif algorithm=='u-stand':
            # 自定义中心标准化，适合数据稳定，变化不大的情况。
            _mean = np.mean(dt)
            _var = np.std(dt)
            _res = (dt - _mean) / _var
        elif algorithm=='decimal':
            # 自定义小数规范化，规范到0~1之间
            _q = np.log10(dt.max())
            _res = dt / np.power(10,_q)
        
        if _custom==True:
            return _res
        else:
            take_data = scaler.fit_transform(dt)
            return take_data
    
    def _mean_poor(self,xs):
        # 计算平均差值,xs:1d
        _mean = np.mean(xs)
        _std = 0
        for j in xs:
            _std += abs(j - _mean)
        _npx = np.sort(xs)
        return (_std / len(xs),_npx[-1] - _npx[0])
        
    def discrete(self,dt,algorithm='auto',n_class=None,scope_val=None,mean_multip=1.2,replace='label'):
        """对数据进行离散化
        args:
        dt:1d;
        algorithm: auto(自动适应),scope(指定范围值划分);
        scope_val: algorithm为scope时使用，指定划分区域，如[5,10,20]=>(x<5,5<=x<10,10<=x<20)
        mean_multip: 与衡量子样的值相乘，值偏小可以细分群体。
        """
        assert len(np.shape(dt))==1,'query data 1d'
        _copy_data = dt.copy()
        _class_dict = dict()
        _class_index = 0
        _sort = np.sort(dt)
        _res = []
        
        def _class_val(_x):
            _q = False
            for d in _class_dict:
                if _x in _class_dict[d]:
                    _q = d
                else:
                    continue
            return _q
        
        def _division():
            # 查找对应值所在的类
            for c in dt:
                _val = _class_val(c)
                if _val==False:
                    raise ValueError('{} not in scope_val'.format(c))
                # 判断用什么值来取代该值。
                if replace=='label':
                    _res.append(int(_val))
                elif replace=='max':
                    _res.append(np.max(_class_dict[_val]))
                elif replace=='min':
                    _res.append(np.min(_class_dict[_val]))
                elif replace=='mean':
                    _res.append(np.mean(_class_dict[_val]))
                else:
                    raise ValueError('{} is dont support function'.format(replace))
        
        
        if algorithm=='auto':
            _before_index = 0
            _before_distant = _sort[1] - _sort[0]
            
            for i in range(1,len(_sort)):
                _div = _sort[i] - _sort[i - 1]
                
                # 若当前值与上一个值之差大于mean_multip * _before_distant，则将之前的值聚为一类。
                if _div >= (mean_multip * _before_distant):
                    od = self._mean_poor(_sort[_before_index:i])[1]
                    if od < _div:
                        _class_dict[str(_class_index)] = _sort[_before_index:i]
                        _before_index = i
                        _class_index += 1
                    
                _before_distant = _div
            
            _class_dict[str(_class_index)] = _sort[_before_index:]

        elif algorithm=='scope':
            # 指定具体的阈值来划分
            assert scope_val!=None,'query scope_val'
            _scope = np.sort(scope_val)
            _pd_val = pd.DataFrame({'a':dt})
            for i,c in enumerate(_scope):
                if i==0:
                    _class_dict[str(_class_index)] = list(_pd_val['a'][_pd_val['a']<=c].values)
                else:
                    _class_dict[str(_class_index)] = list(_pd_val['a'][(_pd_val['a']>_scope[i-1])&(_pd_val['a']<=c)].values)
                _class_index += 1
        # 开始离散化
        _division()
        _classes = [int(i) for i in _class_dict]       
        return (_res,_classes)




#   数据探索性分析
class DataExplorAnalysis(object):
    # 要求输入的数据都是dataFrame形式的数据。
    def __init__(self,data):
        self._model = 'DataExplorAnalysis'
        """self._feature_data_type:
        {
            'feature1':0, #0：定类型数据、1：定序、2：定距、3：定比。
        }
        """
        self._feature_data_type = {}
        self._alter_data(data)

    def _alter_data(self,data):
        self._columns = data.columns
        val = [None for i in range(len(self._columns))]
        self._info = dict(zip(list(self._columns),val))
        self._data = data
    
    @property
    def DataFrame(self):
        return self._data
    @DataFrame.setter
    def DataFrame(self,nd:'DataFrame'):
        self._data = nd
        self._alter_data(nd)

    def _base(self):
        _describe = self._data.describe()

        for i in self._columns:
            self._info[i] = dict(_describe[i])
    
    def _deviateValue(self,dt,mean):
        """求偏态系数
        dt:[2.5,3,...];
        mean:scalar;
        """
        _molecular = 0
        _denominator = 0
        _num = len(dt)

        for i in dt:
            _molecular += np.power((i - mean),3)
            _denominator += np.power((i - mean),2)
        
        _molecular = _molecular / _num
        _denominator = np.power((_denominator / _num),1.5)

        return _molecular / _denominator
    
    def _kurtosisValue(self,dt,mean):
        """求峰态系数
        
        """
        _molecular = 0
        _denominator = 0
        _num = len(dt)

        for i in dt:
            _molecular += np.power((i - mean),4)
            _denominator += np.power((i - mean),2)
        
        _molecular = _molecular / _num
        _denominator = np.power((_denominator / _num),2)

        return _molecular / _denominator

    def _addProperty(self):
        # 添加方差、偏态系数等
        for i in self._columns:
            self._info[i]['var'] = np.power(self._info[i]['std'],2)
            self._info[i]['deviate'] = self._deviateValue(self._data[i],self._info[i]['mean'])
            self._info[i]['kurtosis'] = self._kurtosisValue(self._data[i],self._info[i]['mean'])
            self._info[i]['median'] = np.median(self._data[i])
            self._info[i]['mode'] = stats.mode(self._data[i])[0][0]
    
    def _all_info(self,data=None):
        # 计算各特征属性，data传入的话会重值统计信息。
        if data!=None:
            self._alter_data(data)
        self._base()
        self._addProperty()
        return self._info
    
    def count_info(self,is_print=True,save=False,columns=None,filename="count_info.csv"):
        """# 将各属性信息打印出来，保存为文件
        args: 
        is_print：是否在控制台打印。
        save：是否保存文文件。
        filename：文件路径。
        columns：要打印的特征项，默认是全部。
        """
        self._all_info()
        P = 0.05
        _columns = columns if columns!=None else self._columns

        _infos = dict()

        for c in _columns:
            _deviateTip = "负偏态，大值多在右侧" if self._info[c]['deviate']<0 else "正偏态，大值多在左侧"
            _kurtosisTip = "不是正太分布" if abs(self._info[c]['kurtosis'] - 3) >2 else "近似正太分布"
            
            if self._info[c]['count'] <= 2000:
                _w = stats.shapiro(self._data[c])
                _tip = "近似正态分布(近似度：{})".format(_w[0]) if _w[0]>0.5 and _w[1]>P else "不是正态分布"
            else:
                _ks = stats.kstest(rvs=self._data[c],cdf='norm')
                _tip = "近似正态分布(近似度：{})".format(1 - _ks[0]) if _ks[0]<0.5 and _ks[1]>P else "不是正态分布"
            
            _infos[c] = [self._info[c]['max'],
                         self._info[c]['min'],
                         self._info[c]['mean'],
                         self._info[c]['deviate'],
                         self._info[c]['kurtosis'],
                         self._info[c]['median'],
                         self._info[c]['mode'],
                         _tip,
                         self._info[c]['count'],
                         self._info[c]['var']]
            
            if is_print==True:
                print("#####\t{}:".format(c))
                print('数据条数:\t{}'.format(self._info[c]['count']))
                print('最大值:\t{}'.format(self._info[c]['max']))
                print('最小值:\t{}'.format(self._info[c]['min']))
                print('均值:\t{}'.format(self._info[c]['mean']))
            
                print('偏态系数:\t{}({})'.format(self._info[c]['deviate'],_deviateTip))
                print('峰态系数:\t{}({})'.format(self._info[c]['kurtosis'],_kurtosisTip))
                print('中位数:\t{}'.format(self._info[c]['median']))
                print('众数:\t{}'.format(self._info[c]['mode']))
                
                print('正态性检验结果:\t' + _tip + '\n')
            
        if save==True:
            _data_info = pd.DataFrame(_infos,index=['最大值','最小值','均值','偏态系数','峰态系数','中位数','众数','正态性','数据量','方差'])
            _data_info.to_csv(filename)

            

    def _check_int_nums(self,dt:'[1,2,...]'):
        _int_num = 0
        _repeat_num = 0
        _repeats_arr = []
        _data = dt[0:2000] if len(dt)>2000 else dt
        
        for b in _data:
            if b in _repeats_arr:
                _repeat_num += 1
            else:
                _repeats_arr.append(b)

            if b % 1==0:
                _int_num += 1
            else:
                continue
        # 整数个数、重复数据个数。
        return [_int_num,_repeat_num]

    def analysis_feature_type(self):
        # 大体的判断各特征项数据类型。
        self._feature_data_type = {}
        for i in self._columns:
            _res = self._check_int_nums(self._data[i].values)
            # 整型数大于1/4的数据且重复数据大于1/5就认为是离散型数据。
            if _res[0] > self._info[i]['count'] // 4 and _res[1] > self._info[i]['count'] // 5:

                self._feature_data_type[i] = 1
            else:
                self._feature_data_type[i] = 3

    def _command_fn(self,type1,type2):
        # 根据数据类型推荐使用相关性计算方法
        if (type1==0 and type2 < 2) or (type2==0 and type1 < 2):
            return ['mutualInfo']
        elif type1==1 and type2==1:
            return ['spearman','kendal']
        elif (type1==1 and type2 > 1) or (type2==1 and type1 > 1):
            return ['pearson']
        elif type1 > 1 and type2 > 1:
            return ['pearson','cov']
        else:
            return [None]

    def relatedAnalysis(self,columns=None,save=False,save_path='relation_array.csv'):
        from selfTool import usual_learn
        # 相关性分析，生成相关性矩阵。

        _columns = columns or self._columns
        self._relationArray = np.zeros([len(_columns),len(_columns)],dtype=float)
        _model = usual_learn.RelationCalc()

        for a,f1 in enumerate(_columns):
            for b,f2 in enumerate(_columns):
                if a==b:
                    self._relationArray[a,b] = 1
                    continue
                else:
                    _fn = self._command_fn(self._feature_data_type[f1],self._feature_data_type[f2])
                    if _fn[0]==None:
                        self._relationArray[a,b] = None
                    elif len(_fn)==1:
                        #c = _model.calc(self._data[f1],self._data[f2],fn_str=_fn[0])
                        #print('c===>>>',c)
                        self._relationArray[a,b] = _model.calc(self._data[f1],self._data[f2],fn_str=_fn[0])
                    else:
                        _q = 0.6 * _model.calc(self._data[f1],self._data[f2],fn_str=_fn[0]) + \
                                                     0.4 * _model.calc(self._data[f1],self._data[f2],fn_str=_fn[1])

                        self._relationArray[a,b] = _q
                                                     
        _save_data = pd.DataFrame(self._relationArray,columns=_columns,index=_columns)
        if save==True:
            _save_data.to_csv(save_path)

        return _save_data



class DimentionReduce(object):
    _DESCRIBE = '降维分析'
    def __init__(self):
        self._config = {
            "pca":{
                'n_components':'mle',
                'svd_solver':'auto'
            },
            "rsvd":{
                'flip_sign':False,
                'n_components':4,
                'random_state':None
            },
            "tsvd":{
                'n_components':4
            }
        }
    
    def rough(self,dt,target,move=True,columns=None):
        """粗糙集算法
        args:
        dt: pandas数据；
        target: 预测列；
        columns: 用于做考虑范围的特征项；
        move: 是否去除第一次检测到的重复项非target。
        """
        
        assert type(dt)==pd.core.frame.DataFrame,'dt must be pd.core.frame.DataFrame'
        _feature_dict = dict()
        TOTAL_ITEM = len(dt.index)
        # 先取出所有重复的行
        dt.drop_duplicates(inplace=True)
        _columns = columns or dt.columns
        _columns.remove(target)
        # 除target列外还有重复的列
        _repeat_index = dt[_columns][dt[_columns].duplicated()].index
        # 用于研究的行
        _g_index = list(set(dt.index) - set(_repeat_index))
        
        # 计算没个属性的重要性
        def _calc_repeat(val):
            kc = list(_columns).copy()
            kc.remove(val)
            # 不能区分的条数
            can_use = dt[kc][dt[kc].duplicated()].index
            _feature_dict[val] = len(can_use) / TOTAL_ITEM
        
        _n_column = _columns.copy()
        for c in _columns:
            _calc_repeat(c)
        
        if move==True:
            d = dt.drop(index=_repeat_index)
            d.reset_index()
        
        return _feature_dict
        
    def excellentScale(self):
        # 最优尺度分析
        pass
    
    def reduce(self,x,algorithm='pca',n_components='mle'):
        from sklearn.decomposition import PCA,FactorAnalysis,TruncatedSVD,randomized_svd
        
        if algorithm=='pca':
            _method = PCA(n_components=n_components,copy=True,svd_solver='auto')
            _res = _method.fit_transform(x)
        elif algorithm=='factor':
            # 因子分析
            _method = FactorAnalysis(n_components=n_components)
            _res = _method.fit_transform(x)
        elif algorithm=='rsvd':
            _res = randomized_svd(M=x,n_components=4)
        elif algorithm=='tsvd':
            _method = TruncatedSVD(n_components=3,n_iter=4)
            _res = _method.fit_transform(x)
        
        return _res
      



# 专用于处理nlp数据
class Nlp_data(object):
    """docstring for read_file"""
    # sequence:文件队列；typ：读取的是中文还是英文en;coording_num:线程数
    def __init__(self):
        self.vocab = {}
        self.words_to_id_res = []
        self.words_id = {}
        self._vocab_size = 0
    
    @property
    def vocab_size(self):
      return self._vocab_size
    
    @property
    def wid(self):
      return self.words_id
    @wid.setter
    def wid(self,v):
      self.words_id = v

    @property
    def vb(self):
      return self.vocab
    @vb.setter
    def vb(self,val):
      self.vocab = val

    # 获取：词转id的结果
    @property
    def converse_res(self):
        return self.words_to_id_res

    @converse_res.setter
    def converse_res(self,val):
        self.words_to_id_res = val      

    # 分词、去除停用词、返回词列表，传入一段文字
    def word_cut(self,text,typ='cn',file_path=r'/home/wcs/item/selfTool/resource/cn_stop.txt',stop=False):
      """args:
      text:一段文字；
      typ:语言类型，en：英文;
      file_path:对应语言的停用词文件位置;
      stop:是否做停用词处理；
      """
      import re

      regs = [re.compile('\n'),re.compile('\s'),re.compile(' ')]
      for rs in regs:
        text = re.sub(rs,'',text)

      all_words = []
      if typ=='cn':
        all_words = list(jieba.cut(text))
      else:
        all_words = nltk.word_tokenize(text)
    
      # 去除停用词
      if stop:
        all_words = common.stop_word(all_words,typ=typ,file_path=file_path)
      return all_words

    # 统计每个词的词频,若需要建立多个词表时需要先清空self.vocab
    def calc_wordsNum(self,words_list):
        for word in words_list:
            if word in self.vocab:
                self.vocab[word] = self.vocab[word] + 1
            elif re.search('\d',word):
                continue
            else:
                self.vocab[word] =  1
    # 直接将整个数据矩阵放入来统计,words:2dim=>[sentence,words]
    def count_all(self,words):
      for w in words:
        self.calc_wordsNum(w)
    
    # 剔除小于指定词频的词
    def rid_words(self,vocab=None,min_count=10):
        words = vocab or self.vocab
        words['<pad>'] = 9999
        words['<s>'] = 9999
        words['</s>'] = 9999
        words['<unk>'] = 9999
        words['<mask>'] = 9999
        rid_after = {w:words[w] for w in words if words[w]>=min_count}

        self._vocab_size = len(rid_after.keys())
        self.vocab = rid_after
        return rid_after

    # 根据词频表生成词-id，id-词文件
    def c_wid(self,vocab=None,create_path='',save=False):
        _vocab = vocab or self.vocab
        words = list(_vocab.keys())
        vocab_len = len(words)
        idxs = list(range(vocab_len))

        w_id = dict(zip(words,idxs))
        id_w = dict(zip(idxs,words))

        #<pad>符号位置 置0，符合习惯。
        _last = id_w[0]
        w_id[_last] = w_id['<pad>']
        id_w[w_id['<pad>']] = _last
        w_id['<pad>'] = 0
        id_w[0] = '<pad>'

        self.words_id = w_id
        print('vocab_size:',len(w_id.keys()))
        if save:
          opFile.op_file(create_path+'/words_id.json',data=w_id,method='save')
          opFile.op_file(create_path+'/id_words.json',data=id_w,method='save')
        
        return w_id,id_w

    
    # 将词转换为id,在循环中使用时让back=True，words_llist为1d,back为False时words_list为2d
    def word_to_id(self,words_list,vocab=None,back=True):
        vocab_dict = vocab or self.words_id

        type_list = [list,tuple,np.ndarray]
        _wl = lambda wlst: [vocab_dict.get(w,vocab_dict['<unk>']) for w in wlst]

        if type(words_list[0]) in type_list:
            res = []
            for w1 in words_list:
                if back==False:
                    self.words_to_id_res.append(_wl(w1))
                else:
                    res.append(_wl(w1))
            return res
        else:
            if back==False:
                self.words_to_id_res.append(_wl(words_list))
            else:
                return _wl(words_list)

    # 去除为空的数据
    def drop_empty(self,x,y):
        xd = []
        yd = []
        for a,b in zip(x,y):
            if len(a)==0 or len(b)==0:
                continue
            else:
                xd.append(a)
                yd.append(b)
        return xd,yd
    # 查看最小最大序列长度
    def seq_len(self,ids):
      ls = [len(c) for c in ids]
      return {'min':min(ls),'max':max(ls)}
        


#求两点间距离
def point_distance(x,y):
    a = x if type(x)==np.ndarray else np.array(x)
    b = y if type(y)==np.ndarray else np.array(y)

    c = np.sum(np.square(a-b))
    return np.sqrt(c)


#打乱数据
def shuffle_data(x,y):
    lg = np.arange(0,len(y))
    np.random.shuffle(lg)

    if type(x)!=np.ndarray:
      x = np.array(x)
      y = np.array(y)
    
    res_x = x[lg]
    res_y = y[lg]
    return (res_x,res_y)

#将数据集划分为训练集和测试集
def divide_data(x,y,val=0.8):
    lg = len(y)
    train_len = round(lg*0.8)
    test_len = lg - train_len

    data = {
        "train_x":x[0:train_len],
        "test_x":x[-test_len:],
        "train_y":y[0:train_len],
        "test_y":y[-test_len:]
    }
    return data


#批量读取数据的迭代器。
def get_batch(data,data_num=None,batch=1):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)
    j = 0
    #最后一个迭代项不足batch数时也能使用
    for c in range(iter_numebr):
        batch_res = [i[j:j+batch] for i in data]
        j = j + batch
        yield batch_res


#用于训练rnn网络的数据的特别batch
def rnn_batch(data,batch=1,start_token='',end_token=''):
    data_len = len(data[0])
    iter_numebr = math.ceil(data_len/batch)

    tp = [list,tuple,np.ndarray]
    #未对齐的数据也能使用
    j = 0
    #最后一个迭代项不足batch数时也能使用 
    for c in range(iter_numebr):
        _x_batch = data[0][j:j+batch]
        _y_batch = data[1][j:j+batch]

        x_batch = [[start_token] + pad + [end_token] for pad in _x_batch]
        y_batch = [[start_token] + pad + [end_token] for pad in _y_batch]

        x_sequence = [np.shape(s)[0] for s in x_batch]
        
        if type(y_batch[0]) in tp:
            y_sequence = [np.shape(s)[0] for s in y_batch]
        else:
            y_sequence = [0 for c in range(batch)]
        
        batch_res = [x_batch,y_batch,x_sequence,y_sequence]

        j = j + batch
        yield batch_res

#填充每条数据的序列数到指定长
def padding(data,seq_num,pad=0):
    """
    data:>=2d
    """
    dt = []

    for i,ct in enumerate(data):
        q = seq_num - len(ct)
        if q >= 0:
            v = [j for j in ct]
            for t in range(q):
                v.append(pad)
            dt.append(v)
        else:
            # 若超出指定长度则截断。
            dt.append(ct[0:seq_num])

    return dt


"""
生成一个和原矩阵一样形状的指定值矩阵，
对未对齐的矩阵使用,只支持2d数据
"""
def fill(data,val=0):
  tp = [list,tuple,np.ndarray]
  res = []
  for i in data:
    n = [val for t in i]
    res.append(n)
  return res


"""用百科词向量将词转为词向量
data:must be 2d,
place:未找到词使用的占位
"""
def baiki_vector(data,baiki_path,module='bin',place=0):
  import gensim
  if module=='bin':
    model = gensim.models.KeyedVectors.load_word2vec_format(baiki_path,binary=True)
  else:
    model = gensim.models.Word2Vec.load(baiki_path)
  
  #存储已查找过的词，重复的则不必再到模型中查找。
  buffer_vector = {}
  res = []

  for sen in data:
    sws = []
    for w in sen:
      if w in buffer_vector:
        sws.append(buffer_vector[w])
      else:
        nk = model[w] if w in model.wv.index2word else place
        sws.append(nk)
        buffer_vector[w] = nk
    sws = [place] if len(sws)==0 else sws
    res.append(sws)
  
  empty_vector = [0 for i in range(128)]
  take = Dispose()
  take.miss = [empty_vector,[],0,None]
  dt = take.interpolate(res,axis=0,method='mean',place=place)

  return dt



# 按照词表id顺序构建一个vector词表，用于embedding_lookup
def embedding_vector(vocab,module='baiki',module_path=None):
    import gensim
    assert module_path!=None,'query module_path of vector file'
    vocab_size = len(vocab.keys())
    vocab_embed = [0 for c in range(vocab_size)]

    if module=='baiki':
        empty = [0 for i in range(128)]
        #model = gensim.models.Word2Vec.load(module_path)
        model = gensim.models.KeyedVectors.load_word2vec_format(module_path,binary=True)
        for key in vocab:
            if key in model.wv.index2word:
                vocab_embed[vocab[key]] = list(model[key])
            else:
                vocab_embed[vocab[key]] = empty
    else:
        pass

    return list(vocab_embed)




#将腾讯的词向量文件分成9个json文件
def divide_tencent_vector(tencent_path):
    file = open(tencent_path,'r',encoding='utf-8')
    save_path = "data/tencent_vector"
    WORDER_NUM = 1000000
    #1000000
    vector = []
    i = 5
    j = 0
    idx = 0

    tencent = {
        "f1":{},
        "f2":{},
        "f3":{},
        "f4":{},
        "f5":{},
        "f6":{},
        "f7":{},
        "f8":{},
        "f9":{},
        "f10":{}
    }

    for item in file:
        if j>WORDER_NUM*4:
            arr = item.split()
            arr_one = arr.pop(0)
            #部分一行有两个中文词
            one = len(arr) - 200
            for n in range(one):
                arr_one = arr.pop(0)

            vector = list(map(float,arr))
                
            idx = "f" + str(i)
            tencent[idx][arr_one] = vector
        else:
            pass
 
        if i < 9:
            if j>WORDER_NUM*i:
                save_name = save_path + str(i) + ".json"
                with open(save_name,'w') as jsonObj:
                    json.dump(tencent[idx],jsonObj)

                    i = i + 1
                del tencent[idx]
            else:
                pass       
        else:
            pass
        j = j + 1

    last_name = save_path + str(i) + ".json"
    with open(last_name,'w') as last:
        json.dump(tencent[idx],last)




def tencent_vector(vocab,tencent_path,save_path='',save=True):
    """根据vocab查找腾讯词向量存为数组，转为embedding_look_up()使用的词向量矩阵。
    args:
    vocab:dict json file;
    tencent_path:腾讯词向量存放的文件夹；
    save_path:保存的文件夹；
    save:是否保存。
    """
    VECTOR_SIZE = 200

    tencent_file = {
        "t1":1,
        "t2":2,
        "t3":3,
        "t4":1,
        "t5":2,
        "t6":3,
        "t7":1,
        "t8":2,
        "t9":3,
    }

    _embedd = [0 for i in range(len(vocab.keys()))]
    _pad = [0 for j in range(VECTOR_SIZE)]
    _unk = [random.uniform(-6,0) for i in range(VECTOR_SIZE)]

    # 读取tencent词向量文件。
    def read_tencent(ord):
        path = tencent_path+'/tc'+ str(ord) +'.json'
        with open(path,'r') as f:
            data = json.load(f)
        return data

    _embedd[vocab['<pad>']] = _pad
    _embedd[vocab['<unk>']] = _pad
    for v in ['<s>','</s>','<mask>']:
        _embedd[vocab[v]] = [random.uniform(-10,-1) for c in range(VECTOR_SIZE)]


    del vocab['<s>'],vocab['</s>'],vocab['<pad>'],vocab['<unk>'],vocab['<mask>']

    #逐个打开腾讯文件
    for f in range(1,10):
        key = 't' + str(f)

        tencent_file[key] = read_tencent(f)
        mated_words = []

        for w in vocab:
            if w in tencent_file[key]:
                mated_words.append(w)
                _embedd[vocab[w]] = tencent_file[key][w]
            else:
                continue
        # 删除已经查找到的词。        
        for w2 in mated_words:
            del vocab[w2]
        #销毁文件解除内存占用
        del tencent_file[key]
    
    # 所有未查找到的用unk填充。
    for c in vocab:
        _embedd[vocab[c]] = _unk

    if save:
        opFile.op_file(file_path=save_path + '/embedding.pkl',data=_embedd,model='pkl',method='save')
    else:
        return _embedd


