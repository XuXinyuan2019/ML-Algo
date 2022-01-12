import numpy as np
import sys

def load_train_data(train):
    data = []
    sen = []
    with open(train) as f:
        for i in f.readlines():
            item = i.replace("\n", "").split("\t")
            if len(item)==1:
                data.append(sen)
                sen = []
            else:
                sen.append(item)
        data.append(sen)
    return data

# load index_to_tag.txt and index_to_word.txt to 2 list
def load_index_data(itt, itw):
    with open(itt) as f:
        tag = [i.replace("\n", "") for i in f.readlines()]
    with open(itw) as f:
        word = [i.replace("\n", "") for i in f.readlines()] 
    return tag, word   

def hmm(data, tag, word):
    init = np.ones(len(tag))
    emit = np.ones((len(tag), len(word)))
    tran = np.ones((len(tag), len(tag)))
    for i in data:
        init[dic_tag[i[0][1]]]+=1
        for k, j in enumerate(i):
            emit[dic_tag[j[1]],dic_word[j[0]]]+=1
            if k>0:
                tran[dic_tag[i[k-1][1]],dic_tag[i[k][1]]]+=1
    init = init/sum(init)
    emit = (emit.T/sum(emit.T)).T
    tran = (tran.T/sum(tran.T)).T
    return init, emit, tran

if __name__ == '__main__':
    data = load_train_data(sys.argv[1]) 
    tag, word = load_index_data(sys.argv[3], sys.argv[2])
    dic_tag = {}
    for i, item in enumerate(tag):
        dic_tag[item]=i
    dic_word = {}
    for i, item in enumerate(word):
        dic_word[item]=i
    init, emit, tran = hmm(data, tag, word)
    np.savetxt(sys.argv[4], init) 
    np.savetxt(sys.argv[5], emit) 
    np.savetxt(sys.argv[6], tran) 