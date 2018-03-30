#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:39:50 2018

@author: akinyilmaz
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inp,deriv=False):
    fx = 1.0/(1+np.exp(-inp))
    if deriv == True:
        return fx*(1-fx)
    else:
        return fx

def tanh(inp,deriv=False):
    fx = (np.exp(inp)-np.exp(-inp)) / (np.exp(inp)+np.exp(-inp))
    if deriv == True:
        return 1-fx*fx
    else:
        return fx
    
def relu2(inp,deriv=False):
    k1 = 0.01
    k2 = 1
    up = 500
    fxplus = k2*inp*(inp>0)*(inp<up)
    fxplus = fxplus + np.max(fxplus)*(inp>=up)
    fxminus = k1*inp*(inp<0)
    fx = fxplus + fxminus
    if deriv == True:
        return (inp<up)*(inp>0)*k2 + (inp<=0)*k1
    else:
        return fx

def relu(inp,deriv=False):
    fx_minus = 0.01*inp*(inp<0)
    fx_plus = inp*(inp>=0)
    fx = fx_minus + fx_plus
    if deriv == True:
        return (inp>=0)*1 + (inp<0)*0.01
    else:
        return fx

def lin(inp,deriv=False):
    fx = inp
    if deriv == True:
        return np.ones(shape=fx.shape)
    else:
        return fx


def clip(delta, threshold):
    delta[delta > threshold] = threshold
    delta[delta < -threshold] = -threshold
    return delta


dataset = pd.read_csv('digits.csv',sep=',').values
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=0)


train_features = train_set[:,1:]
#train_features_noisy = train_features + 128*np.random.rand(train_features.shape[0],train_features.shape[1])
train_features_noisy = train_features
train_features_noisy[train_features_noisy > 255.] = 255.
train_results = train_set[:,1:]

test_features = test_set[:,1:]
#test_features_noisy = test_features + 128*np.random.rand(test_features.shape[0],test_features.shape[1])
test_features_noisy = test_features
test_features_noisy[test_features_noisy > 255.] = 255.
test_results = test_set[:,1:]


train_features_noisy = train_features_noisy/255
test_features_noisy = test_features_noisy/255


X_train = train_features_noisy
Y_train = train_features

X_test = test_features_noisy
Y_test = test_features


neuron_inp_size = X_train.shape[1]
neuron_hidden1_size = 700
neuron_hidden2_size = 600
neuron_hidden3_size = 600
neuron_hidden4_size = 700
neuron_out_size = X_train.shape[1]



W1 = np.random.randn(neuron_inp_size,neuron_hidden1_size)*0.01
W2 = np.random.randn(neuron_hidden1_size,neuron_hidden2_size)*0.01
W3 = np.random.randn(neuron_hidden2_size,neuron_hidden3_size)*0.01
W4 = np.random.randn(neuron_hidden3_size,neuron_hidden4_size)*0.01
W5 = np.random.randn(neuron_hidden4_size,neuron_out_size)*0.01
b1 = np.zeros(shape=(1,neuron_hidden1_size))
b2 = np.zeros(shape=(1,neuron_hidden2_size))
b3 = np.zeros(shape=(1,neuron_hidden3_size))
b4 = np.zeros(shape=(1,neuron_hidden4_size))
b5 = np.zeros(shape=(1,neuron_out_size))



nb_epoch = 400
batch_size = 2048
learning_rate0 = 5.e-5


train_errors = []
test_errors = []
rates = []
epochs = []



for epoch in range(1,nb_epoch+1): 
    learning_rate = learning_rate0 - (((epoch-1)**2)/(nb_epoch**2))*learning_rate0
    #learning_rate = learning_rate0
    for batch in range(0,X_train.shape[0],batch_size):
        if X_train.shape[0]-batch < batch_size:
            l_inp = X_train[batch:,:]
            Y_real = Y_train[batch:,:]
        else:
            l_inp = X_train[batch:batch+batch_size,:]
            Y_real = Y_train[batch:batch+batch_size,:]


        s_hidden1 = np.dot(l_inp,W1) + b1
        l_hidden1 = relu(s_hidden1)
        s_hidden2 = np.dot(l_hidden1,W2) + b2
        l_hidden2 = relu(s_hidden2)
        s_hidden3 = np.dot(l_hidden2,W3) + b3
        l_hidden3 = relu(s_hidden3)
        s_hidden4 = np.dot(l_hidden3,W4) + b4
        l_hidden4 = relu(s_hidden4)
        
        s_out = np.dot(l_hidden4,W5) + b5
        l_out = lin(s_out)
        
        dloss = 2*(l_out-Y_real)
        
        l_out_error = (dloss*lin(s_out,deriv=True))/(l_inp.shape[0])
        W5_delta = np.dot(l_hidden4.T,l_out_error)
        b5_delta = np.sum(l_out_error,axis=0).reshape(1,-1)
        
        l_hidden4_error = np.dot(l_out_error,W5.T)*relu(s_hidden4,deriv=True)
        W4_delta = np.dot(l_hidden3.T,l_hidden4_error)
        b4_delta = np.sum(l_hidden4_error,axis=0).reshape(1,-1)
        
        l_hidden3_error = np.dot(l_hidden4_error,W4.T)*relu(s_hidden3,deriv=True)
        W3_delta = np.dot(l_hidden2.T,l_hidden3_error)
        b3_delta = np.sum(l_hidden3_error,axis=0).reshape(1,-1)
        
        l_hidden2_error = np.dot(l_hidden3_error,W3.T)*relu(s_hidden2,deriv=True)
        W2_delta = np.dot(l_hidden1.T,l_hidden2_error)
        b2_delta = np.sum(l_hidden2_error,axis=0).reshape(1,-1)
                 
        l_hidden1_error = np.dot(l_hidden2_error,W2.T)*relu(s_hidden1,deriv=True)
        W1_delta = np.dot(l_inp.T,l_hidden1_error)
        b1_delta = np.sum(l_hidden1_error,axis=0).reshape(1,-1)
        
        
      
        W5 -= learning_rate*clip(W5_delta,10.0)
        b5 -= learning_rate*clip(b5_delta,10.0)
        W4 -= learning_rate*clip(W4_delta,10.0)
        b4 -= learning_rate*clip(b4_delta,10.0)
        W3 -= learning_rate*clip(W3_delta,10.0)
        b3 -= learning_rate*clip(b3_delta,10.0)
        W2 -= learning_rate*clip(W2_delta,10.0)
        b2 -= learning_rate*clip(b2_delta,10.0)
        W1 -= learning_rate*clip(W1_delta,10.0)
        b1 -= learning_rate*clip(b1_delta,10.0)
    
    
    Y_hat_train = lin(np.dot(relu(np.dot(relu(np.dot(relu(np.dot(relu(np.dot(X_train,W1)+b1),W2)+b2),W3)+b3),W4)+b4),W5)+b5)
    train_error = np.sum(np.square(Y_hat_train-Y_train))/(Y_train.shape[0]*2)
    
      
    Y_hat_test = lin(np.dot(relu(np.dot(relu(np.dot(relu(np.dot(relu(np.dot(X_test,W1)+b1),W2)+b2),W3)+b3),W4)+b4),W5)+b5)
    test_error = np.sum(np.square(Y_hat_test-Y_test))/(Y_test.shape[0]*2)

    train_errors.append(train_error)
    test_errors.append(test_error)
    rates.append(learning_rate)
    epochs.append(epoch)



    print("epoch: "+ str(epoch))
    print("learning rate: "+ str(learning_rate))
    print("train error: "+ str(train_error))
    print("test error: " + str(test_error))
    print("-------------------------------------")
    



 
plt.plot(epochs,train_errors,'-b',label="training cost")
plt.plot(epochs,test_errors,'-r',label="test cost")
plt.xlabel("# of iterations")
plt.ylabel("cost")
plt.legend()
plt.grid()
plt.show()          



indice = 300
image_array = (Y_test[indice,:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys')


image_array = (255*test_features_noisy[indice,:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys')

image_array = (Y_hat_test[indice,:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys')
"""
np.savetxt(fname="W1.txt",X=W1)
upload = np.loadtxt(fname="W1.txt",dtype=float)"""


