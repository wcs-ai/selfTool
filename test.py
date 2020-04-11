import sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import file as fl
import re

a = re.split(r'。|；|！|？','发动机岁离开。佛教的？附近的柯！附近的柯林；女郎经过。')
print(a)
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