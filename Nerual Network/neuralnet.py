import sys
import numpy as np
import math

def model(data_in,data_out):
    data = np.loadtxt(data_in,delimiter=",")
    Y_o = data[:,0]
    Y = np.eye(4)[list(map(int,Y_o))].T
    X = data[:,1:].T
    X = np.row_stack((np.ones(X.shape[1]),X))

def output_matrics(outfile):
    outfile = open(outfile, "w", encoding="utf8")
    for i in range(epoch):
        outfile.write("epoch={} crossentropy(train): {}\n".format(i+1, train_loss[i]))
        outfile.write("epoch={} crossentropy(validation): {}\n".format(i+1, val_loss[i]))
    outfile.write("error(train): {}".format(train_err)+'\n')
    outfile.write("error(test): {}".format(val_err))

def initial(iflag, d0, d1):
    if iflag == 1: #random
        weight = np.random.uniform(-0.1, 0.1, (d0, d1-1))
        param = np.column_stack((np.zeros(d0),weight))
    elif iflag == 2: #zero
        param = np.zeros((d0, d1))
    return param

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def loss(Y_hat,Y):
    sum = 0.0
    for i in range(Y.shape[1]):
        sum += -np.dot(Y[:,i],np.log(Y_hat[:,i]))
    return sum/Y.shape[1]
def cross(y_hat,y):
    return -np.dot(y,np.log(y_hat))
    
def error(Y,result):
    err = 0
    for i in range(len(result)):
        if Y[i] != result[i]:
            err += 1
    err = err/len(Y)
    return err

if __name__ == '__main__':

    train_in = sys.argv[1] #'D:/CMU/ML/HW5/data/small_train.csv' 
    val_in = sys.argv[2]#'D:/CMU/ML/HW5/data/small_val.csv' 

    train_out = sys.argv[3]#'D:/CMU/ML/HW5/data/small_train_out.labels' 
    val_out = sys.argv[4]#'D:/CMU/ML/HW5/data/small_val_out.labels' #
    metrics_out = sys.argv[5]#'D:/CMU/ML/HW5/data/metrics_out.txt' 

    data = np.loadtxt(train_in,delimiter=",")
    Y_o = data[:,0]
    Y = np.eye(4)[list(map(int,Y_o))].T
    X = data[:,1:].T
    X = np.row_stack((np.ones(X.shape[1]),X))

    datav = np.loadtxt(val_in,delimiter=",")
    Y_ov = datav[:,0]
    Yv = np.eye(4)[list(map(int,Y_ov))].T
    Xv = datav[:,1:].T
    Xv = np.row_stack((np.ones(Xv.shape[1]),Xv))

    epoch = int(sys.argv[6])#2
    units = int(sys.argv[7])#4
    iflag = int(sys.argv[8])#2
    rate = float(sys.argv[9])#0.1
    ep = 1e-5


    alpha = initial(iflag, units, X.shape[0])
    salpha = initial(2,units, X.shape[0])
    beta = initial(iflag, 4, units+1) #dimension of y is 4
    sbeta = initial(2, 4, units+1) #dimension of y is 4
    train_loss = []
    val_loss = []

    for i in range(epoch):
        for i, x in enumerate(X.T):
            a = np.dot(alpha,x)
            Z = sigmoid(a)
            Z = np.array([1]+list(Z))
            b = np.dot(beta, Z)
            Y_hat = softmax(b)
            
            dldb = Y_hat - Y[:,i]
            dldb = np.array([list(dldb)])
            dldbeta = np.dot(dldb.T,np.array([list(Z)])) 
            dldz = np.dot(dldb,beta[:,1:])
            dlda = np.multiply(np.multiply(dldz,Z[1:]),[1]*(len(Z)-1)-Z[1:]) 
            dldalpha = np.dot(dlda.T,np.array([list(x)])) 
            
            salpha += np.multiply(dldalpha,dldalpha)
            alpha -= np.multiply(rate / ((salpha+ep)**0.5) ,dldalpha)
            sbeta += np.multiply(dldbeta,dldbeta)
            beta -= np.multiply(rate / ((sbeta+ep)**0.5) ,dldbeta)
            
        a = np.dot(alpha, X)
        Z = sigmoid(a)
        Z = np.row_stack((np.ones(Z.shape[1]),Z))
        b = np.dot(beta, Z)
        Y_hat = softmax(b)
        result = [np.argmax(y) for y in Y_hat.T]
        train_loss.append(loss(Y_hat,Y))
        #print("train loss: ", loss(Y_hat,Y))

        a = np.dot(alpha, Xv)
        Z = sigmoid(a)
        Z = np.row_stack((np.ones(Z.shape[1]),Z))
        b = np.dot(beta, Z)
        Y_hatv = softmax(b)
        resultv = [np.argmax(y) for y in Y_hatv.T]
        val_loss.append(loss(Y_hatv,Yv))
        #print("val loss: ", loss(Y_hatv,Yv))

    train_err = error(Y_o,result)
    val_err = error(Y_ov,resultv)
    #print("train error: ", error(Y_o,result))
    #print("val error: ", error(Y_ov,resultv))


    output_matrics(metrics_out)
    np.savetxt(train_out, result, fmt="%d", delimiter="\n")
    np.savetxt(val_out, resultv, fmt="%d", delimiter="\n")