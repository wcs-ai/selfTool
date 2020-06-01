import sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import file as fl
import re

a = re.split(r'。|；|！|？','发动机岁离开。佛教的？附近的柯！附近的柯林；女郎经过。')

#parser_pdf_file('/home/wcs/data/vv.pdf')

"""
import numpy as np
import requests,json
from SPARQLWrapper import SPARQLWrapper, JSON
 
sparql = SPARQLWrapper("http://localhost:3030/knowledge_graph/sparql")
sparql.setQuery(
    SELECT ?object
    WHERE {
        ?红色食品 ?用途 ?object
    }
    LIMIT 5
)
 
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
 
for result in results["results"]["bindings"]:
    print(result)

"""
"""
import csv

fl = open('/home/wcs/data/knowledge_data/ownthink_v2.csv','r')
knowledge = csv.reader(fl)

with open('/home/wcs/data/entity.csv','w') as et,open('/home/wcs/data/relation.csv','w') as rt:
    et_writer = csv.writer(et)
    rt_writer = csv.writer(rt)

    et_writer.writerow([':ID','name',':LABEL'])
    rt_writer.writerow([':START_ID','name',':END_ID',':TYPE'])

    i = 0
    j = 0
    entity_dict = {}

    def _struct(x):
        global j
        if x in entity_dict:
            return entity_dict[x],False
        else:
            _ent = f'entity{j}'
            entity_dict[x] = _ent
            j += 1
            return _ent,True

    for rw in knowledge:
        if rw=='\x000':
            continue
        if len(rw) < 3:
            continue 
        
        start_entity,bl1 = _struct(rw[0])
        end_entity,bl2 = _struct(rw[2])

        if bl1 != False:
            et_writer.writerow([start_entity,rw[0],'ENTITY'])
        if bl2 != False:
            et_writer.writerow([end_entity,rw[2],'ENTITY'])

        rt_writer.writerow([start_entity,rw[1],end_entity,'RELATIONSHIP'])

        if i > 1000:
            break

        i += 1

from py2neo import Database,Graph,RelationshipMatcher,Node,Relationship
#db = Database('http://127.0.0.1:7474')
graph = Graph("http://127.0.0.1:7474",username="neo4j",password="neo4j")

nodes = graph.nodes

n = nodes.match('ENTITY')
m = nodes.match('ENTITY',name='红色食品')
#v = graph.run('match(n:ENTITY{name:"红色食品"}) where n return n')
#v = graph.run('MATCH (a:ENTITY{name:"红色食品"}) RETURN a.name LIMIT 4').data()
v = RelationshipMatcher(graph)

dt = graph.run('MATCH(n:ENTITY{name:"红色食品"})-[q:RELATIONSHIP]-(m:ENTITY{name:"全部人群"}) RETURN *').data()
a = Node("ENTITY",name="红色食品")
b = Node("ENTITY",name="全部人群")
jk = graph.match(list(m),"RELATIONSHIP")

for i in jk:
    #print(i)
    print(i.start_node['name'],i.end_node['name'],dict(i))
"""


###         特征选择
from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel
import pandas as pd
import scipy.stats as ss
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


"""
df = pd.DataFrame({
    "A":ss.norm.rvs(size=10),
    "B":ss.norm.rvs(size=10),
    "C":ss.norm.rvs(size=10),
    "D":np.random.randint(low=0,high=2,size=10)})

x = df.loc[:,["A","B","C"]]
y = df.loc[:,["D"]]
skb = SelectKBest(k=2)
skb.fit(x,y)
m = skb.transform(x)

rfe = RFE(estimator=SVR(kernel="linear"),n_features_to_select=2,step=1)#step是每次迭代减少的特征数
rfe.fit_transform(x,y)

sfm = SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.1)
"""
x = pd.DataFrame({"a":[1,3,4,0,None,2],"b":[0,0,None,3,None,17],"c":[2,7,7,5.3,4.1,2]})


from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

a = pd.Series(data=[[1,2,3],[4,5,6]],index=['a','b'])

b = pd.DataFrame([[1,2],[5,6],[5.5,6.2],[6.5,4.3],[8.2,1.9],[7,0]],columns=['a','b'])

#print(x[['b','c']][0])
#print(x.loc[[0,1]][['a','c']])
print(x['c'].describe())