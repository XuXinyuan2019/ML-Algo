#input file
import csv
def tsv_data(infile):
    infile_data = []
    infile = open(infile,'r')
    infile_content = csv.reader(infile,delimiter='\t')
    for content in infile_content:
        infile_data.append(content)
    return infile_data

#calculate entropy of the Nth attribute(e.g. H(Y):entropy(data,-1))
import math
def entropy(input_data, n):
    #the number of that feature
    dict_n = {}
    for line in input_data:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n
    H = 0.0
    for key,value in dict_n.items():
        H -= value/sum(dict_n.values()) * math.log(value/sum(dict_n.values()),2)
    return H

def dict_n_sorted(input_data, n):
    dict_n = {}
    for line in input_data:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n
    dict_n_sorted_value = sorted(dict_n.items(), key=lambda item:item[1], reverse=True) #dict sorted by value 
    if len(dict_n_sorted_value)==2 and dict_n_sorted_value[0][1]==dict_n_sorted_value[1][1]:
        dict_n_sorted = sorted(dict_n_sorted_value, key=lambda item:item[0], reverse=True) #dict sorted by value 
    else:
        dict_n_sorted = dict_n_sorted_value
    return dict_n_sorted

#calculate H(Y|X) on Nth attribute
def HYX(input_data, n):
    dict_n = {}
    for line in input_data:
        key = line[n]
        dict_n[key] = dict_n.get(key, 0) + 1
    #define a dict of that node: dict_n={'y':30}

    if len(dict_n)==2:
        dict_n_sorted = sorted(dict_n.items(), key=lambda item:item[1], reverse=True) #dict sorted by value 
        left = dict_n_sorted[0][0]
        #define left and right node

        dict_n_left = {}
        dict_n_right = {}
        for line in input_data:
            key = line[-1] #y
            if (line[n]==left):
                dict_n_left[key] = dict_n_left.get(key, 0) + 1
            else:
                dict_n_right[key] = dict_n_right.get(key, 0) + 1
        #define a dict of y(on the condition of that node'value==left/right): dict_left/right_y

        H_left = 0.0
        for key,value in dict_n_left.items():
            H_left -= value/sum(dict_n_left.values()) * math.log(value/sum(dict_n_left.values()),2)
        H_right = 0.0
        for key,value in dict_n_right.items():
            H_right -= value/sum(dict_n_right.values()) * math.log(value/sum(dict_n_right.values()),2)

        HYX = dict_n_sorted[0][1]/(dict_n_sorted[0][1]+dict_n_sorted[1][1])*H_left + dict_n_sorted[1][1]/(dict_n_sorted[0][1]+dict_n_sorted[1][1])*H_right
        return HYX
    else:
        return 1

def IYX(input_data, n):
    return entropy(input_data,-1)-HYX(input_data,n)

def output_labels(result, filename):
    outfile = open(filename, "w", encoding="utf8")
    #for i in result[:-1]:
    for i in result:
        outfile.write(i+'\n')
    #outfile.write(result[-1])
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

class Node():
    def __init__(self, index, father, leftvalue, leftnode, rightvalue, rightnode):
        self.index = index
        self.father = father
        self.leftvalue = leftvalue
        self.leftnode = leftnode
        self.rightvalue = rightvalue 
        self.rightnode = rightnode

        self.leftdata = []
        self.rightdata = []
        self.pre_index = []

def Tree(input_data, root_node, depth, depth_max):

    if depth == depth_max:
        if input_data == root_node.leftdata:
            for i in range(len(node_list)):
                if node_list[i].leftdata == root_node.leftdata:
                    node_list[i].leftnode = dict_n_sorted(input_data,-1)[0][0]
                    break
        elif input_data == root_node.rightdata:
            for i in range(len(node_list)):
                if node_list[i].rightdata == root_node.rightdata:
                    node_list[i].rightnode = dict_n_sorted(input_data,-1)[0][0]
                    break
        return

    #print('a new node with data length',len(input_data),',index:', root_node.index, ', father:', root_node.father)#########
    IYX_list = []
    for i in range(len(input_data[:][0])-1):
        if i not in root_node.pre_index and i != root_node.index:
            IYX_list.append(IYX(input_data, i))
        else: 
            IYX_list.append(-1)
    if input_data == root_node.leftdata:
        for i in range(len(node_list)):
            if node_list[i].leftdata == root_node.leftdata:
                node_list[i].leftnode = IYX_list.index(max(IYX_list))
                break
    elif input_data == root_node.rightdata:
        for i in range(len(node_list)):
            if node_list[i].rightdata == root_node.rightdata:
                node_list[i].rightnode = IYX_list.index(max(IYX_list))
                break

    new_root_node = Node(index = IYX_list.index(max(IYX_list)), father = root_node.index, leftvalue = None, leftnode = None, rightvalue = None, rightnode = None)
    new_root_node.pre_index = [i for i in root_node.pre_index]
    new_root_node.pre_index.append(root_node.index)

    dict_n = dict_n_sorted(input_data, new_root_node.index)
    #print(dict_n)########

    if len(dict_n)==2:

        new_root_node.leftvalue = dict_n_sorted(input_data, new_root_node.index)[0][0]
        new_root_node.rightvalue = dict_n_sorted(input_data, new_root_node.index)[1][0]
        #define left and right node

        for line in input_data:
            if line[new_root_node.index] == new_root_node.leftvalue:
                new_root_node.leftdata.append(line) 
            elif line[new_root_node.index] == new_root_node.rightvalue:
                new_root_node.rightdata.append(line) 

        depth += 1

        if len(set(i[-1] for i in new_root_node.leftdata)) == 1:
            new_root_node.leftnode = list(set(i[-1] for i in new_root_node.leftdata))[0]
            node_list.append(new_root_node)#
        elif len(set(i[-1] for i in new_root_node.leftdata)) == 2:
            node_list.append(new_root_node)#
            Tree(new_root_node.leftdata, new_root_node, depth, depth_max)

        if len(set(i[-1] for i in new_root_node.rightdata)) == 1:
            new_root_node.rightnode = list(set(i[-1] for i in new_root_node.rightdata))[0]
            node_list.append(new_root_node)#
        elif len(set(i[-1] for i in new_root_node.rightdata)) == 2:
            node_list.append(new_root_node)#
            Tree(new_root_node.rightdata, new_root_node, depth, depth_max)
    
    elif len(dict_n)==1:
        if input_data == root_node.leftdata:
            for i in range(len(node_list)):
                if node_list[i].leftdata == root_node.leftdata:
                    node_list[i].leftnode = dict_n_sorted(input_data,-1)[0][0]
                    break
        elif input_data == root_node.rightdata:
            for i in range(len(node_list)):
                if node_list[i].rightdata == root_node.rightdata:
                    node_list[i].rightnode = dict_n_sorted(input_data,-1)[0][0]
                    break


def predict(line, root_node, node_list):
    value = None
    if line[root_node.index] == root_node.leftvalue:
        #print('root_nodes left node',root_node.leftnode)
        if root_node.leftnode in dict_y: #list of y 
            value = root_node.leftnode
        else:
            for node in node_list:
                if node.index == root_node.leftnode and node.father == root_node.index and sorted(node.leftdata+node.rightdata)==sorted(root_node.leftdata):
                    return predict(line, node, node_list)

    elif line[root_node.index] == root_node.rightvalue:
        #print('root_nodes right node',root_node.rightnode)
        if root_node.rightnode in dict_y: #list of y 
            value = root_node.rightnode
        else:
            for node in node_list:
                if node.index == root_node.rightnode and node.father == root_node.index and sorted(node.leftdata+node.rightdata)==sorted(root_node.rightdata):
                    return predict(line, node, node_list)
    return value






import sys
if __name__ == '__main__':
    train_file = sys.argv[1]
    train_data = tsv_data(train_file)

    test_file = sys.argv[2]
    test_data = tsv_data(test_file)

    depth_max = int(sys.argv[3])
    node_list = []

if depth_max == 0:
    majority = dict_n_sorted(train_data[1:],-1)[0][0]
    root_node_0 = Node(index = 0, father = -1, leftvalue = dict_n_sorted(train_data[1:],0)[0][0], leftnode = majority, rightvalue =  dict_n_sorted(train_data[1:],0)[1][0], rightnode = majority)


else:
    depth = 0
    IYX_list = [IYX(train_data[1:], i) for i in range(len(train_data[:][0])-1)]

    root_node_0 = Node(index = IYX_list.index(max(IYX_list)), father = -1, leftvalue = None, leftnode = None, rightvalue = None, rightnode = None)

    node_list.append(root_node_0)
    depth += 1

    dict_y = sorted([dict_n_sorted(train_data[1:],-1)[0][0],dict_n_sorted(train_data[1:],-1)[1][0]]) #sorted list of y, when tie, should choose dict_y[1]


    root_node_0.leftvalue = dict_n_sorted(train_data[1:], root_node_0.index)[0][0]
    root_node_0.rightvalue = dict_n_sorted(train_data[1:], root_node_0.index)[1][0]
    #define left and right node



    for line in train_data[1:]:
        if line[root_node_0.index] == root_node_0.leftvalue:
            root_node_0.leftdata.append(line) 
        elif line[root_node_0.index] == root_node_0.rightvalue:
            root_node_0.rightdata.append(line) 

    if len(set(i[-1] for i in root_node_0.leftdata)) == 1:
        root_node_0.leftnode = list(set(i[-1] for i in root_node_0.leftdata))[0]
    elif len(set(i[-1] for i in root_node_0.leftdata)) == 2:
        Tree(root_node_0.leftdata, root_node_0, depth, depth_max)

    if len(set(i[-1] for i in root_node_0.rightdata)) == 1:
        root_node_0.rightnode = list(set(i[-1] for i in root_node_0.rightdata))[0]
    elif len(set(i[-1] for i in root_node_0.rightdata)) == 2:
        Tree(root_node_0.rightdata, root_node_0, depth, depth_max)


    train_result = []
    for line in train_data[1:]:
        train_result.append(predict(line, root_node_0, node_list))
    output_labels(train_result,sys.argv[4])
    test_result = []
    for line in test_data[1:]:
        test_result.append(predict(line, root_node_0, node_list))
    output_labels(test_result,sys.argv[5])
    output_matrics(train_result,test_result,train_data,test_data,sys.argv[6])

