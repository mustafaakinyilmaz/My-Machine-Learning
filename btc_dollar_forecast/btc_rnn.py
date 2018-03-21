#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:48:43 2018

@author: akinyilmaz
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def tanh(inp,deriv=False):
    fx = (np.exp(inp) - np.exp(-inp))/(np.exp(inp) + np.exp(-inp))
    if deriv == True:
        return 1-fx*fx
    else:
        return fx

def lin(inp,deriv=False):
    fx = inp
    if deriv == True:
        return np.ones(fx.shape)
    else:
        return fx


def sigmoid(inp,deriv=False):
    fx = 1.0/(1+np.exp(-inp))
    if deriv == True:
        return fx*(1-fx)
    else:
        return fx

def relu(inp,deriv=False):
    fx = inp * (inp > 0)
    if deriv == True:
        return (inp>0)*1
    else:
        return fx
    

def clip_gradient(delta, threshold):
    delta[delta > threshold] = threshold
    delta[delta < -threshold] = -threshold
    return delta


def min_max(array,minv,maxv):
    array_ = np.zeros(shape=array.shape)
    for i in range(array.shape[1]):
        array_[:,i] = (array[:,i] - minv[0,i]) / (maxv[0,i] - minv[0,i])
        
    return array_



delay_time = 40


dataset = pd.read_csv("btc.csv",sep=',')
prices = dataset.iloc[:,-1:].values

X_ = []
for i in range(delay_time):
    X_.append(list(prices[i:i-delay_time]))

X = np.array(X_).reshape(delay_time,-1).T

Y = prices[delay_time:]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20)



max_ = np.max(X_train,axis=0).reshape(1,-1)
min_ = np.min(X_train,axis=0).reshape(1,-1)

X_train = min_max(X_train,min_,max_)
X_test = min_max(X_test,min_,max_)



neuron_inp_size = X_train.shape[1]
neuron_hidden1_size = 30
neuron_hidden2_size = 30
neuron_out_size = 1



W1 = np.random.randn(neuron_inp_size,neuron_hidden1_size)*0.01
W2 = np.random.randn(neuron_hidden1_size,neuron_hidden2_size)*0.01
W3 = np.random.randn(neuron_hidden2_size,neuron_out_size)*0.01
b1 = np.zeros((1,neuron_hidden1_size))
b2 = np.zeros((1,neuron_hidden2_size))
b3 = np.zeros((1,neuron_out_size))

nb_epoch = 1000
batch_size = 128
learning_rate0 = 5.e-4


iterations = []
train_costs = []
test_costs = []
alphas = []



for epoch in range(1,nb_epoch+1):
    learning_rate = learning_rate0 - (((epoch-1)**2)/(nb_epoch**2))*learning_rate0

    for batch in range(0,X_train.shape[0],batch_size):
        
        if X_train.shape[0]-batch < batch_size:
            l_inp = X_train[batch:,:]
            Y_real = Y_train[batch:,:]
        else:
            l_inp = X_train[batch:batch+batch_size,:]
            Y_real = Y_train[batch:batch+batch_size,:]
        
        
        s_hidden1 = np.dot(l_inp,W1) + b1
        l_hidden1 = tanh(s_hidden1)
        
        s_hidden2 = np.dot(l_hidden1,W2) + b2
        l_hidden2 = tanh(s_hidden2)
        
        s_out = np.dot(l_hidden2,W3) + b3
        l_out = lin(s_out)

        
        dloss = 2*(l_out-Y_real)
        #dloss = -(Y_real/(l_out+epsilon))
        
        l_out_error = (dloss*lin(s_out,deriv=True))/(l_inp.shape[0])
        l_out_error = clip_gradient(l_out_error,20.0)
        W3_delta = learning_rate*np.dot(l_hidden2.T,l_out_error)
        b3_delta = learning_rate*np.sum(l_out_error,axis=0).reshape(1,-1)
        
        l_hidden2_error = np.dot(l_out_error,W3.T)*tanh(s_hidden2,deriv=True)
        l_hidden2_error = clip_gradient(l_hidden2_error,20.0)
        W2_delta = learning_rate*np.dot(l_hidden1.T,l_hidden2_error)
        b2_delta = learning_rate*np.sum(l_hidden2_error,axis=0).reshape(1,-1)
        
        l_hidden1_error = np.dot(l_hidden2_error,W2.T)*tanh(s_hidden1,deriv=True)
        l_hidden1_error = clip_gradient(l_hidden1_error,20.0)
        W1_delta = learning_rate*np.dot(l_inp.T,l_hidden1_error)
        b1_delta = learning_rate*np.sum(l_hidden1_error,axis=0).reshape(1,-1)
        
        
        
        W3 -= W3_delta
        b3 -= b3_delta
        W2 -= W2_delta
        b2 -= b2_delta
        W1 -= W1_delta
        b1 -= b1_delta
        
    
    

    Y_hat_train = lin(np.dot(tanh(np.dot(tanh(np.dot(X_train,W1)+b1),W2)+b2),W3)+b3)
    train_error = np.sum(np.square(Y_hat_train-Y_train))/(Y_train.shape[0]*2)
    #train_error = -np.sum(Y_train*np.log(Y_hat_train+epsilon))/(Y_train.shape[0])
    

    
    Y_hat_test = lin(np.dot(tanh(np.dot(tanh(np.dot(X_test,W1)+b1),W2)+b2),W3)+b3) 
    test_error = np.sum(np.square(Y_hat_test-Y_test))/(np.shape(Y_test)[0]*2)


    
    print("epoch: "+ str(epoch))
    print("learning rate: "+ str(learning_rate))
    print("train error: "+ str(train_error))
    print("test error: " + str(test_error))
    print("-------------------------------------")

    
    
    iterations.append((epoch-1)/(nb_epoch))
    train_costs.append(train_error)
    test_costs.append(test_error)
    alphas.append(learning_rate)
    
    

plt.plot(iterations,train_costs,'-b',label="training set")
plt.plot(iterations,test_costs,'-r',label="test set")
plt.xlabel("# of iterations (*"+str(nb_epoch)+")")
plt.ylabel("cost")
plt.legend()
plt.grid()
plt.show()


plt.plot(iterations,alphas,'-b',label="alpha")
plt.xlabel("# of iterations (*"+str(nb_epoch)+")")
plt.ylabel("learning rate")
plt.grid()
plt.show()


l = list(range(Y_test.shape[0]))
plt.plot(l,Y_test,'*b',label="Y test")
plt.plot(l,Y_hat_test,'*r',label="Y hat test")
plt.xlabel("Test Set Samples")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()


days = int(input("Enter how many days:"))

l_i = X[-1,:].reshape(1,-1)

for k in range(days):
    l_i_ = min_max(l_i,min_,max_)
    l_h1 = tanh(np.dot(l_i_,W1) + b1)
    l_h2 = tanh(np.dot(l_h1,W2) + b2)
    l_o = lin(np.dot(l_h2,W3) + b3)
    l_inp_insert = np.append(l_i,l_o)
    l_delete = np.delete(l_inp_insert,0,axis=0)
    l_i = l_delete.reshape(1,-1)

    
print("After "+str(days)+" days 1 btc  --- > "+str(l_o))








