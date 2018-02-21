#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:00:12 2018

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
    

def relu(inp,deriv=False):
    k1 = 0.01
    k2 = 0.5
    up = 20
    fxplus = k2*inp*(inp>0)*(inp<up)
    fxplus = fxplus + np.max(fxplus)*(inp>=up)
    fxminus = k1*inp*(inp<0)
    fx = fxplus + fxminus
    if deriv == True:
        return (inp<up)*(inp>0)*k2 + (inp<=0)*k1
    else:
        return fx

def softmax(inp,deriv=False):
    assert len(np.shape(inp)) == 2
    ex = np.exp(inp)
    sm = ex.sum(axis=1,keepdims=True)
    soft = ex/sm
    if deriv == True:
        return (soft*(1-soft))
    else:
        return soft


dataset = pd.read_csv('train.csv',sep=',').values
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=0)
#test_dataset = pd.read_csv('test.csv',sep=',').values


train_features = train_set[:,1:]
train_results = train_set[:,0].reshape(-1,1)

test_features = test_set[:,1:]
test_results = test_set[:,0].reshape(-1,1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
"""train_features = train_features/255
test_features = test_features/255"""


from sklearn.preprocessing import OneHotEncoder
oneEn = OneHotEncoder(categorical_features=[0])
oneEn2 = OneHotEncoder(categorical_features=[0])

train_results = oneEn.fit_transform(train_results).toarray()
test_results = oneEn.fit_transform(test_results).toarray()



X_train = train_features
Y_train = train_results

X_test = test_features
Y_test = test_results


neuron_inp_size = np.shape(X_train)[1]
neuron_hidden1_size = 500
neuron_hidden2_size = 400
neuron_out_size = 10



W1 = np.random.randn(neuron_inp_size,neuron_hidden1_size)*0.01
W2 = np.random.randn(neuron_hidden1_size,neuron_hidden2_size)*0.01
W3 = np.random.randn(neuron_hidden2_size,neuron_out_size)*0.01
b1 = np.random.randn(1,neuron_hidden1_size)
b2 = np.random.randn(1,neuron_hidden2_size)
b3 = np.random.randn(1,neuron_out_size)

nb_epoch = 250
batch_size = 128
learning_rate0 = 1.e-2

epsilon = 1.e-7

train_accur_list = []
test_accur_list = []
epoch_list = []

for epoch in range(1,nb_epoch+1): 
    learning_rate = learning_rate0 - (((epoch-1)**2)/(nb_epoch**2))*learning_rate0
    for batch in range(0,np.shape(X_train)[0],batch_size):
        if np.shape(X_train)[0]-batch < batch_size:
            l_inp = X_train[batch:]
            Y_real = Y_train[batch:]
        else:
            l_inp = X_train[batch:batch+batch_size]
            Y_real = Y_train[batch:batch+batch_size]


        s_hidden1 = np.dot(l_inp,W1) + b1
        l_hidden1 = relu(s_hidden1)
        
        s_hidden2 = np.dot(l_hidden1,W2) + b2
        l_hidden2 = relu(s_hidden2)
        
        s_out = np.dot(l_hidden2,W3) + b3
        l_out = softmax(s_out)

        
        dloss = 2*(l_out-Y_real)
        #dloss = -(Y_real/(l_out+epsilon))
        
        l_out_error = (dloss*softmax(s_out,deriv=True))/(l_inp.shape[0])
        W3_delta = learning_rate*np.dot(l_hidden2.T,l_out_error)
        b3_delta = learning_rate*np.sum(l_out_error,axis=0).reshape(1,-1)
        
        l_hidden2_error = np.dot(l_out_error,W3.T)*relu(s_hidden2,deriv=True)
        W2_delta = learning_rate*np.dot(l_hidden1.T,l_hidden2_error)
        b2_delta = learning_rate*np.sum(l_hidden2_error,axis=0).reshape(1,-1)
        
        l_hidden1_error = np.dot(l_hidden2_error,W2.T)*relu(s_hidden1,deriv=True)
        W1_delta = learning_rate*np.dot(l_inp.T,l_hidden1_error)
        b1_delta = learning_rate*np.sum(l_hidden1_error,axis=0).reshape(1,-1)
        
        
        W3 -= W3_delta
        b3 -= b3_delta
        W2 -= W2_delta
        b2 -= b2_delta
        W1 -= W1_delta
        b1 -= b1_delta
        
    
    

    Y_hat_train = softmax(np.dot(relu(np.dot(relu(np.dot(X_train,W1)+b1),W2)+b2),W3)+b3)
    train_error = np.sum(np.square(Y_hat_train-Y_train))/(Y_train.shape[0]*2)
    #train_error = -np.sum(Y_train*np.log(Y_hat_train+epsilon))/(Y_train.shape[0])
    
    
    real = np.argmax(Y_train,axis=1)
    predicted = np.argmax(Y_hat_train,axis=1)
    
    
    
    Y_hat_test = softmax(np.dot(relu(np.dot(relu(np.dot(X_test,W1)+b1),W2)+b2),W3)+b3)
    
    test_error = np.sum(np.square(Y_hat_test-Y_test))/(np.shape(Y_test)[0]*2)
    #test_error = -np.sum(Y_test*np.log(Y_hat_test+epsilon))/(Y_test.shape[0])
    
    test_real = np.argmax(Y_test,axis=1)
    test_predicted = np.argmax(Y_hat_test,axis=1)
    

    
    
    correct_train = 0
    correct_test = 0
    for k in range(len(real)):
        if real[k] == predicted[k]:
            correct_train += 1
    
    for h in range(len(test_real)):
        if test_real[h] == test_predicted[h]:
            correct_test += 1
    
    train_accurracy = correct_train /  len(real)
    test_accurracy = correct_test / len(test_real)
    
    epoch_list.append(epoch)
    train_accur_list.append(100*train_accurracy)
    test_accur_list.append(100*test_accurracy)
    
    print("epoch: "+ str(epoch))
    print("learning rate: "+ str(learning_rate))
    print("train error: "+ str(train_error))
    print("train accurracy: "+ str(train_accurracy)+" ("+str(correct_train)+"/"+str(len(real))+")")
    print("test error: " + str(test_error))
    print("test accurracy: "+ str(test_accurracy)+" ("+str(correct_test)+"/"+str(len(test_real))+")")
    print("-------------------------------------")

  
    
plt.plot(epoch_list,train_accur_list,'-b',label="training accurracy")
plt.plot(epoch_list,test_accur_list,'-r',label="test accurracy")
plt.plot(epoch_list,list(np.ones((len(epoch_list)))*99.5),'-g',label="human level accurracy")
plt.xlabel("# of iterations")
plt.ylabel("% accurracy")
plt.legend()
plt.grid()
plt.show()


indice = 6
image_array = scaler.inverse_transform(train_features[indice,:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys')

test_output = softmax(np.dot(relu(np.dot(relu(np.dot(train_features[indice,:],W1)+b1),W2)+b2),W3)+b3)
predicted_result = np.argmax(test_output,axis=1) 
print(predicted_result)


