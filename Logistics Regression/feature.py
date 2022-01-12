import csv
import numpy as np
import sys
   
def format_dic(filepath): #file->dic
    infile = open(filepath,'r')
    dic = {}
    infile_content = csv.reader(infile,delimiter='\n')
    for content in infile_content:
        i = content[0].split()
        if len(i) == 2:
            dic[i[0]] = int(i[1])
        else:
            dic[i[0]] = list(map(float,i[1:]))
    length = len(content[0].split())-1
    return dic, length

def format_data(filepath): #file->2D list
    infile = open(filepath,'r')
    infile_content = csv.reader(infile,delimiter='\n')
    labels = []
    comments = []
    for content in infile_content:
        i = content[0].split('\t')
        labels.append(i[0])
        comments.append(i[1].split())
    return list(map(int,labels)), comments

def model1(datafile, dicfile, output):
    labels, comments = format_data(datafile)
    dic, length = format_dic(dicfile) 
    result = np.zeros((len(labels),len(dic)))
    for i in range(len(labels)):
        for word in comments[i]:
            if word in dic.keys():
                result[i][dic[word]]=1
    labels_array = np.array([labels]).T
    result_all = np.column_stack((labels_array, result))
    np.savetxt(output, result_all, fmt="%d", delimiter="\t")
    return
    
def model2(datafile, dicfile, output):
    labels, comments = format_data(datafile)
    dic, length = format_dic(dicfile) 
    result = np.zeros((1,length))
    for i in range(len(labels)):
        temp = np.zeros(length)
        count = 0
        for word in comments[i]:
            if word in dic.keys():
                temp += np.array(dic[word])
                count += 1
        temp = temp/count
        result = np.row_stack((result,temp))
        labels_array = np.array([labels]).T
    result_all = np.column_stack((labels_array, result[1:]))
    np.savetxt(output, result_all, fmt="%f", delimiter="\t")

if __name__ == '__main__':
    datafile1 = sys.argv[1]
    datafile2 = sys.argv[2]
    datafile3 = sys.argv[3]
    dicfile1 = sys.argv[4]
    dicfile2 = sys.argv[9]
    output1 = sys.argv[5]
    output2 = sys.argv[6]
    output3 = sys.argv[7]
    if int(sys.argv[8]) == 1:
        model1(datafile1, dicfile1, output1)
        model1(datafile2, dicfile1, output2)
        model1(datafile3, dicfile1, output3)
    else:
        model2(datafile1, dicfile2, output1)
        model2(datafile2, dicfile2, output2)
        model2(datafile3, dicfile2, output3)
