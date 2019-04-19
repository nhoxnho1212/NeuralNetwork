#caculate
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def error(x_predict,x):
    return np.sum(((x-x_predict)**2)/2)
    
def Loss(x_predict,x):
    return np.sum(-(x*np.log(x_predict)+(1-x)*np.log(1-x_predict)))

def gradient(o_predict,o,x):
    return np.dot((o_predict-o),x.T)

def FeedForward(x1,x2,w1,w2,w3,bias):
    h1=sigmoid(np.dot(x1.T,w1))
    h2=sigmoid(np.dot(x2.T,w2))

    h12=np.concatenate((np.concatenate((bias,h1.T)),h2.T))

    o_predict=sigmoid(np.dot(h12.T,w3))

    return (o_predict,h12)

def Backpropagation(w1,w2,w3,h,o_predict,o,learning_rate,x1,x2):
    gdnt=gradient(o_predict.T,o.T,h)
    w3=w3 - learning_rate * gdnt.T
    h1= np.array([h[1]])
    h2= np.array([h[2]])

    #and
    AND=np.array([[0,0,0,1]])
    #or
    OR=np.array([[0,1,1,1]])
    w1=w1 - learning_rate * gradient(h1,AND,x1).T
    w2=w2 - learning_rate * gradient(h2,OR,x2).T

    return (w1,w2,w3)

def NN(x,o,learning_rate=1,tol=10e-6,count_max=10000):
    w1=[[0.7],
    [0.5],
    [0.3]]

    w2=[[0.8],
        [0.2],
        [0.1]]

    w3=[[0.9],
        [0.25],
        [0.7]]

    bias=[[1,1,1,1]]

    x1=np.concatenate((bias,x))
    x2=np.concatenate((bias,x))

    #gradient descent
    pre_loss=10e10
    #update weight
    for count in range(count_max):
        (o_predict,h)=FeedForward(x1,x2,w1,w2,w3,bias)
        (w1,w2,w3)=Backpropagation(w1,w2,w3,h,o_predict,o,learning_rate,x1,x2)
        L=Loss(o_predict,o)
        print(L)
        if count%20==0:
            if np.abs(pre_loss-L)<tol:
                print('\ncount: ',count)
                return (w1,w2,w3)
            else:
                pre_loss=L
    print('\ncount: ',count)
    return (w1,w2,w3)

#--------------------------------------------------------------------------------------------
x=np.array([[0,0,1,1],
             [0,1,0,1]])

o=np.array([[0,1,1,0]]).T

#train
learning_rate=0.8
(w1,w2,w3)=NN(x,o,learning_rate)

#result train
print(w1,w1,w3,sep='\n\n')
bias=[[1,1,1,1]]

x1=np.concatenate((bias,x))
x2=np.concatenate((bias,x))

print('\n\no_predict:\n',FeedForward(x1,x2,w1,w2,w3,bias)[0])