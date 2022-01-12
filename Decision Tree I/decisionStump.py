import csv
def tsv_data(infile):
    infile_data = []
    infile = open(infile,'r')
    infile_content = csv.reader(infile,delimiter='\t')
    for content in infile_content:
        infile_data.append(content)
    return infile_data

def modeling(input_data, n):
    #the number of that feature
    dict_n = {}
    for line in input_data[1:]:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n

    dict_n_sorted = sorted(dict_n.items(), key = lambda item:item[1], reverse=True) #dict sorted by value 
    left = dict_n_sorted[0][0]
    right = dict_n_sorted[1][0]
    #define left and right 

    dict_n_left = {}
    dict_n_right = {}
    for line in input_data[1:]:
        key = line[-1]
        if line[n]==left: dict_n_left[key] = dict_n_left.get(key, 0) + 1
        if line[n]==right: dict_n_right[key] = dict_n_right.get(key, 0) + 1
        
    left_value = max(dict_n_left, key=dict_n_left.get) #get the key to the max value
    right_value = max(dict_n_right, key=dict_n_right.get) #get the key to the max value
    
    tree_n = {left:left_value, right:right_value}

    return tree_n


def predict(tree_n, input_data, n):
    result = []
    for line in input_data[1:]:
        result.append(tree_n[line[n]])
    return result

def output_labels(result, filename):
    outfile = open(filename, "w", encoding="utf8")
    for i in result[:-1]:
        outfile.write(i+'\n')
    outfile.write(result[-1])
    outfile.close()
    return

def error(predict, input_data):
    err = 0
    for i in range(len(predict)):
        if input_data[i+1][-1]!=predict[i]:
            err+=1
    err_rate = err/len(predict)
    return err_rate

def output_matrics(result_train, result_test, train_data, test_data, filename):
    outfile = open(filename, "w", encoding="utf8")
    outfile.write("error(train): {}".format(error(result_train,train_data))+'\n')
    outfile.write("error(test): {}".format(error(result_test,test_data)))
    outfile.close()
    return

import sys
if __name__ == '__main__':
    
    index = int(sys.argv[3]) #0

    train_file = sys.argv[1] #'small_train.tsv'
    train_data = tsv_data(train_file)

    test_file = sys.argv[2] #'small_test.tsv'
    test_data = tsv_data(test_file)

    tree_n = modeling(train_data, index)


    result_train = predict(tree_n, train_data, index)
    output_labels(result_train, sys.argv[4])

    result_test = predict(tree_n, test_data, index)
    output_labels(result_test, sys.argv[5])

    output_matrics(result_train, result_test, train_data, test_data, sys.argv[6])