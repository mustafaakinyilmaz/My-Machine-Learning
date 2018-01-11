#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:37:55 2018

@author: akinyilmaz
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv',sep=',')
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,[0,3,4,5,6,9]])
X[:,[0,3,4,5,6,9]] = imputer.transform(X[:,[0,3,4,5,6,9]])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEnCo = LabelEncoder()
labelEnGen = LabelEncoder()
X[:,1] = labelEnCo.fit_transform(X[:,1])
X[:,2] = labelEnGen.fit_transform(X[:,2])
oneEn = OneHotEncoder(categorical_features=[1])
X = oneEn.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


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


neuron_inp_size = np.shape(X_train)[1]
neuron_hidden_size = 10
neuron_hidden_size2 = 8
neuron_out_size = 1

W1 = np.random.randn(neuron_inp_size,neuron_hidden_size)
W2 = np.random.randn(neuron_hidden_size,neuron_hidden_size2)
W3 = np.random.randn(neuron_hidden_size2,neuron_out_size)

b1 = np.random.randn(neuron_hidden_size,)
b2 = np.random.randn(neuron_hidden_size2,)
b3 = np.random.randn(neuron_out_size,)

batch_size = 4
nb_epoch = 200
learning_rate = 1.e-2

Y_train = np.reshape(Y_train,(np.shape(Y_train)[0],1))



l_whole_inp = X_train.copy()
real_whole_out = Y_train.copy()
    
s_whole_hidden1 = np.dot(l_whole_inp,W1) + b1
l_whole_hidden1 = relu(inp=s_whole_hidden1)
s_whole_hidden2 = np.dot(l_whole_hidden1,W2) + b2
l_whole_hidden2 = relu(inp=s_whole_hidden2)
s_whole_out = np.dot(l_whole_hidden2,W3) + b3
l_whole_out = sigmoid(inp=s_whole_out)
train_mean_error = np.sum(np.square(l_whole_out-real_whole_out))/np.shape(Y_train)[0]
    
# Whole Test Data
l_whole_test = X_test.copy()
real_whole_test_out = np.reshape(Y_test,(np.shape(Y_test)[0],1))
    
s_hidden1_test = np.dot(l_whole_test,W1) + b1
l_hidden1_test = relu(inp=s_hidden1_test)
s_hidden2_test = np.dot(l_hidden1_test,W2) + b2
l_hidden2_test = relu(inp=s_hidden2_test)
s_test_out = np.dot(l_hidden2_test,W3) + b3
l_test_out = sigmoid(inp=s_test_out)
    
test_mean_error = np.sum(np.square(l_test_out-real_whole_test_out))/np.shape(Y_test)[0]

print("-------------------------------------")
print("starting train error: " + str(train_mean_error))
print('starting test error: '+str(test_mean_error))



# Train
cost_list = np.array([],dtype=float)
for epoch in range(1,nb_epoch+1): 
    for batch in range(0,np.shape(X_train)[0],batch_size):
        if np.shape(X_train)[0]-batch < batch_size:
            Y_real = Y_train[batch:]
            l_inp = X_train[batch:]
        else:
            Y_real = Y_train[batch:batch+batch_size]
            l_inp = X_train[batch:batch+batch_size]
        s_hidden1 = np.dot(l_inp,W1) + b1
        l_hidden1 = relu(inp=s_hidden1)
        s_hidden2 = np.dot(l_hidden1,W2) + b2
        l_hidden2 = relu(inp=s_hidden2)
        s_out = np.dot(l_hidden2,W3) + b3
        l_out = sigmoid(inp=s_out)
        
        # Using Squared Error
        l_out_error = np.multiply((l_out - Y_real),sigmoid(s_out,deriv=True))
        W3_delta = learning_rate*np.dot(l_hidden2.T,l_out_error)
        b3_delta = np.sum(l_out_error)
        
        l_hidden2_error = np.multiply(np.dot(l_out_error,W3.T),relu(s_hidden2,deriv=True))
        W2_delta = learning_rate*np.dot(l_hidden1.T,l_hidden2_error)
        b2_delta = np.sum(l_hidden2_error,axis=0)
        
        l_hidden1_error = np.multiply(np.dot(l_hidden2_error,W2.T),relu(s_hidden1,deriv=True))
        W1_delta = learning_rate*np.dot(l_inp.T,l_hidden1_error)
        b1_delta = np.sum(l_hidden1_error,axis=0)
        
        W3 -= W3_delta
        b3 -= b3_delta
        W2 -= W2_delta
        b2 -= b2_delta
        W1 -= W1_delta
        b1 -= b1_delta
     
    
    if epoch > 2:
        momentum = cost_list[0]/cost_list[1]
        learning_rate *= momentum
        cost_list = cost_list[1]
        
    # Whole Training Data
    l_whole_inp = X_train.copy()
    real_whole_out = Y_train.copy()
    
    s_whole_hidden1 = np.dot(l_whole_inp,W1) + b1
    l_whole_hidden1 = relu(inp=s_whole_hidden1)
    s_whole_hidden2 = np.dot(l_whole_hidden1,W2) + b2
    l_whole_hidden2 = relu(inp=s_whole_hidden2)
    s_whole_out = np.dot(l_whole_hidden2,W3) + b3
    l_whole_out = sigmoid(inp=s_whole_out)
    
    train_mean_error = np.sum(np.square(l_whole_out-real_whole_out))/np.shape(Y_train)[0]
    cost_list = np.append(cost_list,train_mean_error)
    
    # Whole Test Data
    l_whole_test = X_test.copy()
    real_whole_test_out = np.reshape(Y_test,(np.shape(Y_test)[0],1))
    
    s_hidden1_test = np.dot(l_whole_test,W1) + b1
    l_hidden1_test = relu(inp=s_hidden1_test)
    s_hidden2_test = np.dot(l_hidden1_test,W2) + b2
    l_hidden2_test = relu(inp=s_hidden2_test)
    s_test_out = np.dot(l_hidden2_test,W3) + b3
    l_test_out = sigmoid(inp=s_test_out)
    
    test_mean_error = np.sum(np.square(l_test_out-real_whole_test_out))/np.shape(Y_test)[0]
    
    
    print("-------------------------------------")
    print("epoch: "+str(epoch))
    print("learning rate: "+str(learning_rate))
    print("train error: " + str(train_mean_error))
    print('test error: '+str(test_mean_error))
    
    
    
    
com_nn = (l_test_out > 0.5)*1
com_real = real_whole_test_out
count = 0
for i in range(np.shape(l_test_out)[0]):
    if com_nn[i,0] == com_real[i,0]:
        count += 1

class_rate = count/np.shape(l_test_out)[0]
print("classification rate: "+str(class_rate))
    

        
        
        



