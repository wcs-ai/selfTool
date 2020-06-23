import sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

class Tree(object):
    def __init__(self):
        """
        每个元素是一个节点表示，分别为：双亲节点指针域、数据域、节点指针域(可以有多个)。
        保证第一个节点是头指针。
        _saveArea:[[parent,{id:1,name:'a',...},node1,node2,...],[],[],...]
        """
        self._tree = []
        self._header = 0
        self._valKey = 'val'
        pass
    
    def _search(self,nid):
        """按照节点id查找节点所在路径"""
        _address = None
        for idx,i in enumerate(self._tree):
            if i[0]['id']==nid:
                _address = idx
        
        return _address
                
    
    def _update(self,node):
        """
        从上而下的更新树：父亲节点id，当前节点数据域、是第几个子节点(从左往右)。
        node:[pNode,node,ord];=>[id,{id:0,name:2,val:1},ord]
        """
        assert len(node)==3,"node's length query 3,but it is {}".format(len(node))
        # 写入当前节点id
        if not node[1]['id']:
            node[1]['id'] = len(self._tree)
        
        if not node[1]['val']:
            raise ValueError('not found val in node[1]')
        
        _padd = self._search(node[0])
        # 链表中加入当前节点
        self._tree.append([_padd,node[1],None])
        # 其父节点添加该子节点指针。
        if (node[2] - len(self._tree[_padd]) - 3) > 0:
            self._tree[_padd].append(ord)
        else:
            self._tree[_padd][2 + ord] = len(self._tree) - 1
            
    def _add(self,node,idx,keyVal):
        """二叉排序树添加节点
        node:{name,id,val:1}
        """
        if not node['id']:
            raise ValueError('query key id')
        
        _t = self._tree[idx]
        if node[keyVal] < _t[1][keyVal]:
            # 与其左节点比较
            if _t[2] and _t[2]!=None:
                # 递归查找满足的位置
                self._add(node,_t[2],keyVal)
            else:
                # 左节点不存在的情况,插入该节点，更新父亲节点左指针。
                self._tree.append([idx,node,None,None])
                self._tree[idx][2] = len(self._tree) - 1
                return len(self._tree) - 1
        elif node[keyVal] > _t[1][keyVal]:
            # 与其右节点比较
            if _t[3] and _t[3]!=None:
                # 递归查找满足的位置
                self._add(node,_t[3],keyVal)
            else:
                # 左节点不存在的情况,插入该节点，更新父亲节点左指针。
                self._tree.append([idx,node,None,None])
                self._tree[idx][3] = len(self._tree) - 1
                return len(self._tree) - 1
        else:
            return False
        
    def _ordTree(self,nodes,keyVal):
        """# 构建二叉排序树
        nodes:[{name:1,val:5},{},...]
        key:指定用该键值比较各节点大小
        """
        _ord = list(sorted(nodes),key=lambda x:x[keyVal],reverse=True)
        
        _middle = len(_ord) // 2
        self._header = 0
        nodes[_middle]['id'] = _middle
        self._tree = [[None,nodes[_middle],None,None]]
        
        for i,c in enumerate(_ord):
            c['id'] = i
            if i==_middle:
                continue
            elif i==0 or i==_middle:
                self._add(c,0,keyVal)
            else:
                # 以后改进一下
                self._add(c,0,keyVal)
                
        
        
        
         
        
        