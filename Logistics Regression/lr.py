import numpy as np 
import sys

def sigmoid(x):
    if x>=0:      
        return 1.0/(1+np.exp(-x))
    else:
        return np.exp(x)/(1+np.exp(x))

def par(x,y,theta):
    result = x * (y-sigmoid(np.dot(theta,x)))
    return result

def SGD(theta,X,Y):
    for i in range(len(X)):
        theta = theta + alpha/len(X)*par(X[i],Y[i],theta)
    return theta
    
def predict(X,theta):
    result = np.zeros(len(X))
    for i in range(len(X)):
        y = np.dot(theta,X[i])
        if y<=0:
            result[i] = 0
        else:
            result[i] = 1
    return result

def error(Y,result):
    err = 0
    for i in range(len(result)):
        if Y[i] != result[i]:
            err += 1
    err = err/len(Y)
    return err

def output_matrics(Y_train,result_train,Y_test,result_test,outfile):
    outfile = open(outfile, "w", encoding="utf8")
    outfile.write("error(train): {}".format(error(Y_train,result_train))+'\n')
    outfile.write("error(test): {}".format(error(Y_test,result_test)))
    outfile.close()
    return

if __name__ == '__main__':
    dicfile = sys.argv[4]
    epoch = int(sys.argv[8])
    alpha = 0.01

    train_in = sys.argv[1]
    vali_in = sys.argv[2]
    test_in = sys.argv[3]

    train_out = sys.argv[5]
    test_out = sys.argv[6]

    data_train = np.genfromtxt(fname=train_in, delimiter="\t") 
    Y_train = data_train[:,0]
    X_train = data_train[:,1:]
    X_train = np.column_stack((np.ones(len(X_train)),X_train))

    theta = np.zeros(len(X_train[0]))
    for i in range(epoch):
        theta = SGD(theta,X_train,Y_train)

    result_train = predict(X_train,theta)

    data_test = np.genfromtxt(fname=test_in, delimiter="\t") 
    Y_test = data_test[:,0]
    X_test = data_test[:,1:]
    X_test = np.column_stack((np.ones(len(X_test)),X_test))

    result_test = predict(X_test,theta)

    np.savetxt(train_out, result_train, fmt="%d", delimiter="\n")
    np.savetxt(test_out, result_test, fmt="%d", delimiter="\n")

    output_matrics(Y_train,result_train,Y_test,result_test,sys.argv[7])