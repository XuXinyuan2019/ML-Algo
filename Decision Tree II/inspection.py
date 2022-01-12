import math
import csv

def tsv_data(infile):
    infile_data = []
    infile = open(infile,'r')
    infile_content = csv.reader(infile,delimiter='\t')
    for content in infile_content:
        infile_data.append(content)
    return infile_data

def entropy(input_data, n):
    #the number of that feature
    dict_n = {}
    for line in input_data[1:]:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n

    H = 0.0
    for key,value in dict_n.items():
        H -= value/sum(dict_n.values()) * math.log(value/sum(dict_n.values()),2)
    return H

def error(input_data, n):
    #the number of that feature
    dict_n = {}
    for line in input_data[1:]:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n

    dict_n_sorted = sorted(dict_n.items(), key = lambda item:item[1], reverse=True) #dict sorted by value 
    err = dict_n_sorted[1][1]/sum(dict_n.values())
    return err

def output_matrics(x, y, filename):
    outfile = open(filename, "w", encoding="utf8")
    outfile.write("entropy: {}".format(x)+'\n')
    outfile.write("error: {}".format(y))
    outfile.close()
    return

import sys
if __name__ == '__main__':
    entr = entropy(tsv_data(sys.argv[1]),-1)
    erro = error(tsv_data(sys.argv[1]),-1)
    output_matrics(entr, erro, sys.argv[2])
    
    