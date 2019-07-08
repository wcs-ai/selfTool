#!/usr/bin/python
#-*-coding:UTF-8-*-
#extract toponymy of article

#B:实体首部，M：实体内部，S：单独构成实体，E：实体尾部，O：不是实体部分
symbol = ['，','、','；','：','？','——']

def toponymy(datas,save_path):
    arr = []
    word = []
    file = open(save_path, 'a')
    for i in datas:
        if i[0] in symbol:
            continue
        elif i[0]=='。':
            arr.append(word)
            file.write('\n')
            word = []
        elif i[1]=='ns':
            #是地名
            w_len = len(i[0])
            if w_len==0:
                file.write(i[0]+' '+i[1]+' S\n')
                word.append((i[0],i[1],'S'))
            else:
                mark = ['B']+['M']*(w_len-2)+['E']
                for j in range(w_len):
                    file.write(i[0][j]+' '+i[1]+' '+mark[j]+'\n')
                    word.append((i[0][j],i[1],mark[j]))
        else:
            a_len = len(i[0])
            tag = ['O']*a_len
            for g in range(a_len):
                word.append((i[0][g],i[1],tag[g]))
                file.write(i[0][g] + ' ' + i[1] + ' ' + tag[g]+'\n')

    file.close()


