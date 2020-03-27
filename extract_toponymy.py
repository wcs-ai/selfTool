import numpy as np
from openpyxl import load_workbook
import csv
import rdflib

xml = open('resource/knowledge.xml','a')

with open('/home/wcs/data/kg_data/ownthink_v2.csv','r') as f:
    reader = csv.reader(f)

    i = 0
    xml.write('<?xml version="1.0"?>\n')
    xml.write('<RDF>\n')
    title_list = {}

    try:
        for rw in reader:
            if rw=='\x00':
                continue

            if rw[0] not in title_list.keys():
                title_list[rw[0]] = []

            title_list[rw[0]].append(rw[1:])

            if i > 15:
                break
            i += 1
    except:
        print(title_list)
    del title_list['实体']
import re
g = rdflib.Graph()
for c in title_list:
    s = rdflib.URIRef(c)
    for t in title_list[c]:
        p = rdflib.URIRef('http://baike.com/resource/'+re.sub('\s','',t[0]))
        o = rdflib.URIRef('http://baike.com/resource/'+re.sub('\s','',t[1]))
        g.add((s,p,o))

g.serialize('resource/hello.rdf',encoding="UTF-8")


{
    '__root__': {
                'b': {'_val': 2,
                     'f': {'_val': 1,
                           'a': {'_val': 1, 
                                 'm': {'_val': 1}
                                }
                          },
                     'c': {'_val': 1, 
                           'd': {'_val': 1,
                                 'f': {'_val': 1}
                                }, 
                           'k': {'_val': 1,
                                 'g': {'_val': 1}
                                }
                          }
                      }, 
                'c': {'_val': 1, 
                      'k': {'_val': 1, 
                            'g': {'_val': 1}
                           }
                     }, 
                'f': {'_val': 1}}, 
    '_val': 1
}



