#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:45:49 2018

@author: akinyilmaz
"""
"""log = np.log(l_out)
mul = -(log*Y_train)
cross_ent = mul.sum(axis=1,keepdims=True)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)


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
neuron_hidden_size = 200
neuron_out_size = 10



W1 = np.random.randn(neuron_inp_size,neuron_hidden_size)
W2 = np.random.randn(neuron_hidden_size,neuron_out_size)
b1 = np.random.randn(neuron_hidden_size,)
b2 = np.random.randn(neuron_out_size,)


batch_size = 50
nb_epoch = 50
learning_rate = 7.e-4

cost_list = np.array([],dtype=float)

for epoch in range(1,nb_epoch+1): 
    for batch in range(0,np.shape(X_train)[0],batch_size):
        if np.shape(X_train)[0]-batch < batch_size:
            l_inp = X_train[batch:]
            Y_real = Y_train[batch:]
        else:
            l_inp = X_train[batch:batch+batch_size]
            Y_real = Y_train[batch:batch+batch_size]


        s_hidden = np.dot(l_inp,W1) + b1
        l_hidden = sigmoid(s_hidden)
        s_out = np.dot(l_hidden,W2) + b2
        l_out = softmax(s_out)


        l_out_error = (l_out - Y_real)*softmax(s_out,deriv=True)
        W2_delta = learning_rate*np.dot(l_hidden.T,l_out_error)
        b2_delta = learning_rate*np.sum(l_out_error,axis=0)
        
        l_hidden_error = np.dot(l_out_error,W2.T)*sigmoid(s_hidden,deriv=True)
        W1_delta = learning_rate*np.dot(l_inp.T,l_hidden_error)
        b1_delta = learning_rate*np.sum(l_hidden_error,axis=0)
        W2 -= W2_delta
        b2 -= b2_delta
        W1 -= W1_delta
        b1 -= b1_delta
        
    
    if epoch > 2:
        momentum = cost_list[0]/cost_list[1]
        learning_rate *= momentum
        cost_list = cost_list[1]
    
    s_whole_hidden = np.dot(X_train,W1) + b1
    l_whole_hidden = sigmoid(s_whole_hidden)
    s_whole_out = np.dot(l_whole_hidden,W2) + b2
    l_whole_out = softmax(s_whole_out)
    train_error = np.sum(np.square(l_whole_out-Y_train))/np.shape(Y_train)[0]
    cost_list = np.append(cost_list,train_error)
    
    real = np.argmax(Y_train,axis=1)
    predicted = np.argmax(l_whole_out,axis=1)
    
    
    
    test_whole_hidden = sigmoid(np.dot(X_test,W1) + b1)
    test_whole_out = softmax(np.dot(test_whole_hidden,W2) + b2)
    
    test_error = np.sum(np.square(test_whole_out-Y_test))/np.shape(Y_test)[0]
    test_real = np.argmax(Y_test,axis=1)
    test_predicted = np.argmax(test_whole_out,axis=1)
    

    
    
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
    
    print("epoch: "+ str(epoch))
    print("learning rate: "+ str(learning_rate))
    print("train error: "+ str(train_error))
    print("train accurracy: "+ str(train_accurracy))
    print("test error: " + str(test_error))
    print("test accurracy: "+ str(test_accurracy))
    print("-------------------------------------")

    

"""test_dataset = pd.read_csv('test.csv',sep=',').values
test_dataset = scaler.transform(test_dataset)
c = input('Want to test? (y/n)')

while c.lower() == 'y':
    indice = np.random.randint(0,np.shape(test_dataset)[0],1)
    print(indice)
    sample = test_dataset[indice]
    image_array = np.reshape(sample,newshape=(28,28))
    
    test_hidden = sigmoid(np.dot(sample,W1) + b1)
    test_output = softmax(np.dot(test_hidden,W2) + b2)
    predicted_test = np.argmax(test_output,axis=1)
    #print("prediction: " + str(predicted_test))
    plt.title('prediction: ' + str(predicted_test))
    plt.imshow(image_array,cmap='Greys')
    time.sleep(2)
    plt.close()"""


    
