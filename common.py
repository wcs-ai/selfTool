#!/usr/bin/python
#-*-coding:UTF-8-*-
import threading
import queue, os
import numpy as np
import math, time
import chardet,copy


#多线程基类
class _Multi_process(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = ''
        self.ord = 0

    def run(self):
        self.ord += 1


#使用多线程的类
class MP(object):
    def __init__(self, num):
        self._threads = []
        self._num = num

    #使用时在start和stop间添加自己想要运行的程序。
    def start(self):
        for i in range(self._num):
            self._threads.append(_Multi_process())
            self._threads[i].start()

    def stop(self):
        for c in self._threads:
            c.join()


#查看字符使用的编码格式
def look_encode(obj):
    v = chardet.detect(obj)
    return v['encoding']


#去除停用词,typ:cn为中文，en为英文
def stop_word(words, typ='cn', file_path='selfTool/cn_stop.txt'):
    #停用词处理，传入数据格式：[(word,v),(),...],纯函数

    # cn_stop = 'SELF_TOOLS/cn_stop.txt'
    # en_stop = 'SELF_TOOLS/en_stop.txt'
    # file_path = cn_stop if typ=='cn' else en_stop
    read = open(file_path, 'rb').read()
    code = chardet.detect(read)
    stop_words = read.decode(code['encoding']).replace('\r', '').split('\n')
    arr = []
    res = ''

    #words是一个列表的情况
    if isinstance(words, list) or isinstance(words, tuple):
        for i in words:
            if isinstance(i, list) or isinstance(i, tuple):
                if i[0] in stop_words:
                    continue
                else:
                    arr.append(i)
            else:
                if i in stop_words:
                    continue
                else:
                    arr.append(i)
    else:
        #传入的是一个字符的情况
        if words in stop_words:
            arr = True
        else:
            arr = False

    return arr

#test the precision about two target
def precision(x, y):
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if isinstance(y, np.ndarray):
        pass
    else:
        y = np.array(y)

    z = list(x == y)
    leg = len(z)
    ok = z.count(True)
    val = round(ok / leg)
    return val


# 去除字符串中的一些换行符。。。
def dislodge(s, alpha=None):
    import re
    _alpha = alpha or ['\n', '\s', ' ', '\o', '\t', '\v', '\f', '\b', '\a']

    for i in _alpha:
        s = re.sub(i, '', s)

    return s


# 批量转为one_hot标签
def one_hot(batch_label, deep=10, gap=0, axis=0):
    #默认类是从0开始
    hot_res = []
    for b in batch_label:
        if axis == 0:
            lb = [1 if (c + gap) == b else 0 for c in range(deep)]
            hot_res.append(lb)
        elif axis == 1:
            batch = []
            for c in b:
                nb = [1 if (c + gap) == q else 0 for q in range(deep)]
                batch.append(nb)
            hot_res.append(batch)
    return hot_res


# 用annoy库对训练好的词向量构建快速查找
def construct_search(path):
    from gensim.models import KeyedVectors
    import json
    from collections import OrderedDict
    tc_wv_model = KeyedVectors.load_word2vec_format(path, binary=True)
    word_index = OrderedDict()
    # counter为索引，key为词;构建一个id词汇映射表，并存为json文件
    """
    for counter,key in enumerate(tc_wv_model.vocab.keys()):
        word_index[key] = counter    

    with open('data/baike.json','w') as fp:
        json.dump(word_index,fp)
    """

    from annoy import AnnoyIndex

    tc_index = AnnoyIndex(128)

    for i, key in enumerate(tc_wv_model.vocab.keys()):
        # tc_wv_model[key]为词对应的词向量
        v = tc_wv_model[key]
        # 每条数据按 (索引,词) 加入
        tc_index.add_item(i, v)

    # 传入的数表示建立的树的个数，多则精度高，但所需时间长
    tc_index.build(30)
    tc_index.save('data/baike_vector.ann')


# 计算卷积，池化中VALID情况想要的卷积核或滤波器宽度。
def cnn_padding(input_width,
                step,
                out_width=None,
                nucel_width=None,
                method="VALID"):
    res_width = ''
    # 计算输出宽度
    if out_width == None:
        if method == "VALID":
            res_width = (input_width - nucel_width + 1) / step
        else:
            res_width = input_width / step

        res_width = math.floor(res_width)
    else:
        # 计算卷积核宽
        if method == "VALID":
            res_width = input_width - (out_width * step - 1)
        else:
            res_width = 0

    return res_width


#发送邮件
"""
{
    info:'dsdfa',
    from:'wu',
    to:'ll',
    title:'',
    email:'19565...'
}
"""


def send_email(info):
    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header
    #罗婷：'2513879704@qq.com'
    sender = "18313746328@qq.com"
    receivers = [info['email']]

    message = MIMEText(info['info'], 'plain', 'utf-8')
    message['From'] = Header(info['from'], "utf-8")  #发件人项
    message['To'] = Header(info['to'], 'utf-8')  #收件人

    message['subject'] = Header(info['title'], 'utf-8')  #邮件标题

    smtpObj = smtplib.SMTP('smtp.qq.com', 25)
    smtpObj.login(sender, "mdmiylsovcthdjfd")
    smtpObj.sendmail(sender, receivers, message.as_string())


#对字典排序
def rank_dict(dc, reverse=True, apply=1):
    #apply:指定依赖于键值排序(0)还是依赖于value值排序(1)
    sort_arr = [
        si[1 - apply]
        for si in sorted(dc.items(), key=lambda k: k[apply], reverse=reverse)
    ]
    return sort_arr


#bm25算法实现
class BM25(object):
    def __init__(self, docs):
        #docs：[[w1,w2],[w1,w2,...],...]
        self.D = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1) /
                      (self.f[index][word] + self.k1 *
                       (1 - self.b + self.b * d / self.avgdl)))
        return score

    #doc:[w1,w2,w3,...];计算当前句与每一句的得分。返回得分列表
    def every_score(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


#HMM算法实现
class HMM(object):
    def __init__(self):
        self.model_file = 'hmm_model.pkl'
        #词首、词中、词尾、单独成词
        self.state_list = ['B', 'M', 'E', 'S']
        self.load_para = False

    #True则导入训练好的模型参数，否则初始化参数用于训练
    def try_load_model(self, trained):
        if trained:
            import pickle
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            #状态转移概率（state->state's condition probability）
            self.A_dic = {}
            #发射概率（state->words' condition probability）
            self.B_dic = {}
            #状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    def train(self, path):
        #重置几个概率矩阵
        self.try_load_model(False)
        #统计各标记出现总次数
        count_dic = {}
        """初始化参数
        将几个标记作为键添加到3个字典中去
        A_dic:{
            "B":{"B":0.0,"M":0.0,"E":0.0,"S":0.0},
            ...
        }
        B_dic:{
            "B":{"字1":0,"字2":1,...}
            ...
        }
        Pi_dic:{
            "B":0.0,
            ...
        }
        """
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                self.Pi_dic[state] = 0.0
                self.B_dic[state] = {}

                count_dic[state] = 0

        #传入的是一个词,返回该词对应的标记
        def makeLabel(text):
            out_text = []
            #长度为1时说明改字单独成词
            if len(text) == 1:
                out_text.append('S')
            else:
                #词长大于2时有词中标记，否则标记为词首，词尾。
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
            return out_text

        init_parameters()
        line_num = -1
        #观察者集合
        words = set()
        #打开训练文件，每行是一句话，每句话是已经切分好的词
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_num += 1
                #无用
                line = line.strip()

                if not line:
                    continue

                #字列表
                word_list = [i for i in line if i != ' ']
                words != set(word_list)  #更新字的集合
                #词列表
                linelist = line.split()
                line_state = []

                for w in linelist:
                    line_state.extend(makeLabel(w))
                #字长与标记长对应
                assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    #统计所有循环中各标记出现的频率
                    count_dic[v] += 1
                    if k == 0:
                        #每个句子的第一个状态用于计算初始概率
                        self.Pi_dic[v] += 1
                    else:
                        """
                        计算转移概率(该标记对应的前一个标记中的对应标记v值加1).对应公式中的p(o)=p(o1)p(o1|o2)*p(o2|o3)...
                        发射概率(当前位置标记对应它的字的分数加1)。对应公式中的p(r|o)=p(r1|o1)*p(r2|o2)*...
                        转移概率公式中有一个初始概率p(o1)，所以需要一个Pi_dic.
                        所有循环完成之后即是对所有训练数据的分数累加。
                        计算出的A_dic表示每个标记对应的后一个各标记的频率。
                        B_dic则记录各标记对应的有哪些字及其频率。
                        """
                        self.A_dic[line_state[k - 1]][v] += 1
                        self.B_dic[line_state[k]][
                            word_list[k]] = self.B_dic[line_state[k]].get(
                                word_list[k], 0) + 1.0
        #只有B,S标记的值不为0.
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        #将频率转化为概率,B_dic中加1做平滑处理
        self.A_dic = {
            k: {k1: v1 / count_dic[k]
                for k1, v1 in v.items()}
            for k, v in self.A_dic.items()
        }
        self.B_dic = {
            k: {k1: (v1 + 1) / count_dic[k]
                for k1, v1 in v.items()}
            for k, v in self.B_dic.items()
        }

        import pickle
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    #viterbi算法
    def viterbi(self, text, states, start_p, trans_p, emit_p):
        """args:
        text:一句话。
        states:self.state_list。/[B,M,S,E]
        start_p:初始概率。/self.Pi_dic
        trans_p:转移概率。/self.A_dic
        emit_p:发射概率。/self.B_dic
        v:[{"E":12.1,"M":10.2,...}]
        path:{"S":["S"],...}
        """
        v = [{}]
        path = {}
        for y in states:
            #初始概率乘以发射概率
            v[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            v.append({})
            newpath = {}
            #检测训练的发射概率中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                        text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and \
                        text[t] not in emit_p['B'].keys()
            """
            未知字单独成词，概率为1.0。
            要计算每个字对应的各个标记的概率，所以需要循环整个标记列表。 
            这里忽略了p(r)的计算,因为下面这个循环中p(r)的值是一样的。
            """
            for y in states:
                #获取对应标记下对应字的发射概率
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                """
                每个标记下对应的各标记的转移概率乘以该字的发射概率，再乘以前一个字的对应各标记的概率。
                每次循环计算前一时刻(字)的不同标记概率乘以当前时刻(字)的不同标记的转移概率，然后求最大概率路径(viterbi思想)。
                p(r|o)*p(o)=p(r1|o1)*p(o1|o2)*...
                以下生成的值为：[(p(r1|o1)*p1*p(o1|o1),o1),
                                (p(r1|o1)*p1*p(o1|o2),o2), 
                                ...
                               ]
                """
                (prob, state) = max([
                    (v[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                    for y0 in states if v[t - 1][y0] > 0
                ])
                #将各个字对应的各个标记的概率添加到数组v中。
                v[t][y] = prob
                """
                因为上式中每次的循环计算是针对y的发射概率和转移概率，所以这里最后加 [y].
                因为选出的最大概率情况只有四种：B,M,E,S。所以path中只用了这四种情况统计四种路径，
                然后每次循环加上当前对应的标记y。
                """
                newpath[y] = path[state] + [y]
            #path中每个路径的列表长度都一样
            path = newpath

        #最后一个字的最大概率值及其标记
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(v[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(v[len(text) - 1][y], y) for y in states])
        #返回选中的最大概率路径
        return (prob, path[state])

    #对传入的句子进行分词
    def cut(self, text):
        if not self.load_para:
            #存在训练好的模型则导入，否则初始化
            self.try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic,
                                      self.A_dic, self.B_dic)
        begin, next_ = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin:i + 1]
                next_ = i + 1
            elif pos == 'S':
                yield char
                next_ = i + 1

        if next_ < len(text):
            yield text[next_:]


# N-Gram算法计算两字符串相似度。
def N_Gram(st1, st2, N=2):
    #按N长切分字符串
    n_cut = lambda st: [st[i:i + N] for i in list(range(len(st)))[::N]]

    #两字符串相似度
    def simlar(s1, s2):
        #列表转为集合，并将各集合内重复值个数做为交集的差集个数。
        js1 = set(s1)
        js2 = set(s2)
        #交集
        intersection = js1 & js2
        #n-gram公式
        score = len(s1) + len(s2) - N * len(intersection)
        return score

    st1 = n_cut(st1)

    scores = []
    # 与多条字符串匹配。
    if isinstance(st2, list):
        for j in st2:
            cut_st2 = n_cut(j)
            scores.append(simlar(st1, cut_st2))
        return scores
    else:
        st2 = n_cut(st2)
        res = simlar(st1, st2)
        return res


# 简单的记录程序运行时间。2020-3-14
class Cord_programmer(object):
    def __init__(self):
        self._time = 0
        self._start_time = time.time()
        self._end_time = 0

    def s_to_t(self, ts):
        _h = math.floor(ts / 60)
        _m = math.floor((ts % 60) / 60)
        _s = math.floor((ts % 60) % 60)
        return _h, _m, _s

    def end(self):
        self._end_time = time.time()
        used_time = self._end_time - self._start_time

        h, m, s = self.s_to_t(used_time)
        print(f'used=>{h} h-{m} m-{s} s,total_seconds:{used_time}')




# 关联分析
class FP_Growth(object):
    def __init__(self, data, min_fp=2):
        """args:
        data:[[a,b,...],[a,c,...]];
        min_fp:各项中的最大频率项如果低于该频率会被剔除。
        """
        self._data = data
        self._alone_fp = {}
        self._min_fp = min_fp
        self._fp_tree = {'__root__': {}, '_val': 1}
        # 头指针表
        self._head_list = []
        # 属性节点列表。
        self._pointer_property = ['_val']
        # 查找结尾路径时使用的变量。
        self._end_paths = []
        # 频繁路径集
        self._fp_paths = {}

        self.fp_coord()
        self.create_head_list()
        self.create_fp_tree()

    @property
    def tree(self):
        return self._fp_tree

    # 为各个单项记录频率。
    def fp_coord(self):
        for i in self._data:
            for j in i:
                if j in self._alone_fp:
                    self._alone_fp[j] += 1
                else:
                    self._alone_fp[j] = 1

    def create_head_list(self):
        sort_data = {}
        for a in self._data:
            fp_list = [self._alone_fp[i] for i in a]
            now_dict = dict(zip(a, fp_list))
            # 剔除设定小的项。
            if max(fp_list) < self._min_fp:
                continue
            else:
                sort_data[sum(fp_list)] = rank_dict(now_dict,reverse=True,apply=1)
        # 各项和总项都排好序的结果。
        self._head_list = rank_dict(sort_data,reverse=True,apply=0)

    def _tree(self, dic, q_father_pointer, n_father_pointer, sun_pointer):
        """args:
        dic:当前局部字典。
        q_father:前提父节点。
        n_father:当前父节点。
        sun_pointer:需要的子节点。
        """

        if q_father_pointer == n_father_pointer:
            # 前提父节点相同时，子节点是否已存在情况。
            if sun_pointer not in dic[q_father_pointer].keys():
                dic[q_father_pointer][sun_pointer] = dict()
                dic[q_father_pointer][sun_pointer]['_val'] = 0
            dic[q_father_pointer][sun_pointer]['_val'] += 1
        else:
            # 当前父节点不同时，遍历其所有子节点。
            for q in dic[n_father_pointer].keys():
                if q not in self._pointer_property:
                    self._tree(dic[n_father_pointer], q_father_pointer, q,
                               sun_pointer)
                else:
                    continue

    def create_tree(self, tree_dict, lt):
        for i, c in enumerate(lt):
            _qf = "__root__" if i == 0 else lt[i - 1]
            self._tree(tree_dict, _qf, "__root__", c)

    # 构建频繁项树。
    def create_fp_tree(self):
        for t in self._head_list:
            self.create_tree(self._fp_tree, t)

    # 遍历fp树，查找所有路径。
    def search_end(self, dic,paths):
        for i in dic:
            if i in self._pointer_property:
                continue
            else:
                self._search(dic[i],i,paths)
        # 遍历完一个节点下的所有路径后回退到上一级。
        if len(paths) > 0:
            paths.pop()

    def _search(self,dic,now_pointer,paths):
        paths.append(now_pointer)
        if len(dic.keys()) == 1:
            self._end_paths.append(copy.deepcopy(paths))
            # 该条路径完成后，删除最后一个路径值，回退到上一级。
            paths.pop()
        else:
            self.search_end(dic,paths)

    # 计算各种组合的频繁项集。
    def _construct_fp_path(self,end_pointer):
        for c in self._end_paths:
            if end_pointer in c:
                _index = c.index(end_pointer)
                now_pointers = c[0:_index]

                for t in now_pointers:
                    if t in self._fp_paths[end_pointer]['pointer_score']:
                        self._fp_paths[end_pointer]['pointer_score'][t] += 1
                    else:
                        self._fp_paths[end_pointer]['pointer_score'][t] = 1

    def end_fps(self):
        max_k = rank_dict(self._alone_fp,reverse=True,apply=1)[0]
        for a in self._alone_fp:
            # 不满足频繁项和是最大频繁指的不用参与构建。
            if self._alone_fp[a] < self._min_fp or a == max_k:
                continue
            else:
                self._fp_paths[a] = {"pointer_score":{}}
                self._construct_fp_path(a)

    def every_tuple_score(self):
        pass



"""
dt = [["a", "b", "d",'c'], ["f", "b", "m",'a','b'], ["f", "b", "c",'d'], ["k", "b", "f",'c'],
      ["k", "g", "c"], ["a", "b", "c",'d'], ["d", "b", "f",'c']]

fp = FP_Growth(dt)
g = []
fp.search_end(fp.tree['__root__'],g)
"""
#print(fp._head_list)
