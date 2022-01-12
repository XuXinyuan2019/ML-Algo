import sys
import numpy as np

def load_valid_data(valid):
    data = []
    sen = []
    with open(valid) as f:
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

def load_valid(valid_data):
    obser_list = []
    Y_list = []
    for i in valid_data:
        a = []
        b = []
        for j in i:
            a.append(j[0])
            b.append(j[1])
        obser_list.append(a)
        Y_list.append(b)
    return obser_list, Y_list

def forward(obser, init, emit, tran):
    alpha = np.zeros((len(obser), len(tag)))
    alpha[0] = np.log(init)+np.log(emit[:,dic_word[obser[0]]])
    for k in range(1,alpha.shape[0]):
        alpha[k] = np.log(emit[:,dic_word[obser[k]]])
        for i in range(alpha.shape[1]):
            v = []
            for j in range(alpha.shape[1]):
                v.append(alpha[k-1][j]+np.log(tran[j][i]))
            m = max(v)
            alpha[k][i] += m
            s = 0
            for j in v:
                s += np.exp(j-m)
            alpha[k][i] += np.log(s)
    return alpha

def backward(obser, emit, tran):
    beta = np.zeros((len(obser), len(tag)))
    beta[-1] = np.zeros(len(tag))
    for k in range(beta.shape[0]-2,-1,-1):
        beta[k] = np.zeros(beta.shape[1])
        for i in range(beta.shape[1]):
            v = []
            for j in range(beta.shape[1]):
                v.append(np.log(emit[j,dic_word[obser[k+1]]]) + beta[k+1][j] + np.log(tran[i][j]))
            m = max(v)
            beta[k][i] += m
            s = 0
            for j in v:
                s += np.exp(j-m)
            beta[k][i] += np.log(s)
    return beta

def FB(alpha, beta):
    return [tag[list(i).index(max(list(i)))] for i in alpha+beta]

def output_matrics(outfile,acc):
    outfile = open(outfile, "w", encoding="utf8")
    outfile.write("Average Log-Likelihood: {}".format(sum(log_list)/len(log_list))+'\n')
    outfile.write("Accuracy: {}".format(acc))

def output_predicted(outfile,obser_list,result_list):
    outfile = open(outfile, "w", encoding="utf8")
    for i in range(len(result_list)):
        for j in range(len(result_list[i])):
            outfile.write(obser_list[i][j]+'\t'+result_list[i][j]+'\n')
        outfile.write('\n')

def errorall(Y_list,result_list):
    err = 0
    cnt = 0
    for i in range(len(result_list)):
        for j in range(len(result_list[i])):
            cnt += 1
            if Y_list[i][j] != result_list[i][j]:
                err += 1
    return err/cnt


if __name__ == '__main__':
    valid_data = load_valid_data(sys.argv[1])
    obser_list, Y_list = load_valid(valid_data)
    tag, word = load_index_data(sys.argv[3], sys.argv[2])
    dic_tag = {}
    for i, item in enumerate(tag):
        dic_tag[item]=i
    dic_word = {}
    for i, item in enumerate(word):
        dic_word[item]=i
    init = np.loadtxt(sys.argv[4])
    emit = np.loadtxt(sys.argv[5])
    tran = np.loadtxt(sys.argv[6])

    result_list = []
    log_list = []

    for i in range(len(obser_list)):
        obser, Y = obser_list[i], Y_list[i]
        alpha = forward(obser, init, emit, tran)
        beta = backward(obser, emit, tran)
        result = FB(alpha, beta)
        result_list.append(result)
        m = max(alpha[-1])
        log = 0
        log += m
        s = 0
        for j in alpha[-1]:
            s += np.exp(j-m)
        log += np.log(s)
        log_list.append(log)
        acc = 1- errorall(Y_list,result_list)

    output_matrics(sys.argv[8],acc)
    output_predicted(sys.argv[7],obser_list,result_list)