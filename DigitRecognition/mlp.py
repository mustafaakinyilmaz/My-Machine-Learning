#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 01:13:24 2018

@author: akinyilmaz
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Define activation functions
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
    k = 0.01
    fxplus = inp*(inp>=0)
    fxminus = k*inp*(inp<0)
    fx = fxplus + fxminus
    if deriv == True:
        return (inp<0)*k + (inp>0)*1
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

"""def clip(delta,threshold):
    delta[delta >= threshold] = threshold
    delta[delta < -threshold] = -threshold
    return delta"""



def create_dataset():
    dataset = pd.read_csv('train.csv',sep=',').values
    from sklearn.model_selection import train_test_split
    # Randomly split dataset into train and test sets
    train_set,test_set = train_test_split(dataset,test_size=0.2)
    
    
    train_features = train_set[:,1:]
    train_results = train_set[:,0].reshape(-1,1)
    
    test_features = test_set[:,1:]
    test_results = test_set[:,0].reshape(-1,1)
    
    # Normalize features
    train_features = train_features/255
    test_features = test_features/255
    
    from sklearn.preprocessing import OneHotEncoder
    oneEn = OneHotEncoder(categorical_features=[0])
    
    # One hot encode digit outputs
    train_results = oneEn.fit_transform(train_results).toarray()
    test_results = oneEn.fit_transform(test_results).toarray()
    
    
    
    X_train = train_features
    Y_train = train_results
    
    X_test = test_features
    Y_test = test_results
    
    return X_train,Y_train,X_test,Y_test

def initialize_parameters(inp_size,h1_size,h2_size,out_size):
    W1 = np.random.randn(inp_size,h1_size)*0.1
    W2 = np.random.randn(h1_size,h2_size)*0.1
    W3 = np.random.randn(h2_size,out_size)*0.1
    b1 = np.zeros(shape= (1,h1_size))
    b2 = np.zeros(shape= (1,h2_size))
    b3 = np.zeros(shape= (1,out_size))
    
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3
                  }
    return parameters


def forward(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    # Add two hidden layers with relu activation
    s_hidden1 = np.dot(X,W1) + b1
    l_hidden1 = relu(s_hidden1)
        
    s_hidden2 = np.dot(l_hidden1,W2) + b2
    l_hidden2 = relu(s_hidden2)
     
    # Add a output layer    
    s_out = np.dot(l_hidden2,W3) + b3
    l_out = softmax(s_out)
    
    hiddens = {"s_hidden1": s_hidden1,
               "s_hidden2": s_hidden2,
               "s_out": s_out
                }
    
    activations = {"l_hidden1": l_hidden1,
                   "l_hidden2": l_hidden2,
                   "l_out": l_out
                    }
    
    return hiddens,activations

def backprop(parameters,hiddens,activations,X,Y):
    W3 = parameters["W3"]
    W2 = parameters["W2"]
    
    s_hidden1 = hiddens["s_hidden1"]
    s_hidden2 = hiddens["s_hidden2"]
    s_out = hiddens["s_out"]
    
    l_hidden1 = activations["l_hidden1"]
    l_hidden2 = activations["l_hidden2"]
    l_out = activations["l_out"]

    # Compute gradients for MSE
    dloss = l_out-Y
    
    l_out_error = (dloss*softmax(s_out,deriv=True))/(X.shape[0])
    W3_delta = np.dot(l_hidden2.T,l_out_error)
    b3_delta = np.sum(l_out_error,axis=0).reshape(1,-1)
        
    l_hidden2_error = np.dot(l_out_error,W3.T)*relu(s_hidden2,deriv=True)
    W2_delta = np.dot(l_hidden1.T,l_hidden2_error)
    b2_delta = np.sum(l_hidden2_error,axis=0).reshape(1,-1)
        
    l_hidden1_error = np.dot(l_hidden2_error,W2.T)*relu(s_hidden1,deriv=True)
    W1_delta = np.dot(X.T,l_hidden1_error)
    b1_delta = np.sum(l_hidden1_error,axis=0).reshape(1,-1)


    deltas = {"W1_delta": W1_delta,
              "W2_delta": W2_delta,
              "W3_delta": W3_delta,
              "b1_delta": b1_delta,
              "b2_delta": b2_delta,
              "b3_delta": b3_delta
                }
    return deltas
    

def update_parameters(parameters,deltas,learning_rate,clip_thr):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    W1_delta = deltas["W1_delta"]
    W2_delta = deltas["W2_delta"]
    W3_delta = deltas["W3_delta"]
    b1_delta = deltas["b1_delta"]
    b2_delta = deltas["b2_delta"]
    b3_delta = deltas["b3_delta"]
    
    # Update parameters after clipping gradients in a certain range
    W3 -= learning_rate*np.clip(W3_delta,-clip_thr,clip_thr)
    W2 -= learning_rate*np.clip(W2_delta,-clip_thr,clip_thr)
    W1 -= learning_rate*np.clip(W1_delta,-clip_thr,clip_thr)
    b3 -= learning_rate*np.clip(b3_delta,-clip_thr,clip_thr)
    b2 -= learning_rate*np.clip(b2_delta,-clip_thr,clip_thr)
    b1 -= learning_rate*np.clip(b1_delta,-clip_thr,clip_thr)

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "b1": b1,
                  "b2": b2,
                  "b3": b3
                  }
    return parameters 


def evaluate(parameters,X,Y):
    # Calculate error, and correct desicion number
    _,activations = forward(X,parameters)
    Y_hat = activations["l_out"]
    error = np.sum(np.square(Y_hat-Y))/(Y.shape[0]*2)
    
    real = np.argmax(Y,axis=1)
    predicted = np.argmax(Y_hat,axis=1)
    
    correct = 0
    for k in range(len(real)):
        if real[k] == predicted[k]:
            correct += 1
            
    #accuracy = correct/len(real)
    
    return error,correct,len(real)


def model(X_train,Y_train,X_test,Y_test,nb_epoch,batch_size,learning_rate0):
    neuron_inp_size = np.shape(X_train)[1]
    # Two hidden layers each having 500 neurons. It can be changed
    neuron_hidden1_size = 500
    neuron_hidden2_size = 500
    neuron_out_size = 10
    
    parameters = initialize_parameters(neuron_inp_size,neuron_hidden1_size,neuron_hidden2_size,neuron_out_size)
    epoch_list = []
    train_accur_list = []
    train_error_list = []
    test_accur_list = []
    test_error_list = []
    
    # Iterations
    for epoch in range(1,nb_epoch+1): 
        learning_rate = learning_rate0 - (((epoch-1)**2)/(nb_epoch**2))*learning_rate0
        #learning_rate = learning_rate0
        for batch in range(0,np.shape(X_train)[0],batch_size):
            if np.shape(X_train)[0]-batch < batch_size:
                l_inp = X_train[batch:,:]
                Y_real = Y_train[batch:,:]
            else:
                l_inp = X_train[batch:batch+batch_size,:]
                Y_real = Y_train[batch:batch+batch_size,:]
            
            hiddens,activations = forward(l_inp,parameters)
            
            deltas = backprop(parameters,hiddens,activations,l_inp,Y_real)
            
            parameters = update_parameters(parameters,deltas,learning_rate,10.0)
            
        
        train_error,train_correct,train_total = evaluate(parameters,X_train,Y_train)
        
        test_error,test_correct,test_total = evaluate(parameters,X_test,Y_test)
        
        train_accuracy = 100*(train_correct/train_total)
        test_accuracy = 100*(test_correct/test_total)
        
        epoch_list.append(epoch)
        train_accur_list.append(train_accuracy)
        train_error_list.append(train_error)
        test_accur_list.append(test_accuracy)
        test_error_list.append(test_error)
        print("epoch: "+ str(epoch))
        print("learning rate: "+ str(learning_rate))
        print("train error: "+ str(train_error))
        print("train accurracy: "+ str(train_accuracy)+" ("+str(train_correct)+"/"+str(train_total)+")")
        print("test error: " + str(test_error))
        print("test accurracy: "+ str(test_accuracy)+" ("+str(test_correct)+"/"+str(test_total)+")")
        print("-------------------------------------")
    
    return parameters,epoch_list,train_accur_list,train_error_list,test_accur_list,test_error_list

X_train,Y_train,X_test,Y_test = create_dataset()

# Training. nb_epoch, batch_size, and learning_rate0 are adjustable.
parameters,epoch_list,train_accur_list,train_error_list,test_accur_list,test_error_list= model(X_train,
                                                                                               Y_train,
                                                                                               X_test,
                                                                                               Y_test,
                                                                                               nb_epoch=30,
                                                                                               batch_size=128,
                                                                                               learning_rate0=2.e-2)
# plot iterations vs MSE
plt.plot(epoch_list,train_error_list,label="training error")
plt.plot(epoch_list,test_error_list,label="test error")
plt.legend()
plt.grid()
plt.xlabel("iterations")
plt.ylabel("MSE")


# plot iterations vs Accuracy
plt.plot(epoch_list,train_accur_list,label="training accuracy")
plt.plot(epoch_list,test_accur_list,label="test accuracy")
plt.legend()
plt.grid()
plt.xlabel("iterations")
plt.ylabel("% Accuracy")

# See sample predictions on test test. Change indice to change samples.
indice = 1
test_1 = X_test[indice,:]
_,act = forward(test_1,parameters)
pred = np.argmax(act["l_out"],axis=1)

plt.imshow(test_1.reshape((28,28)),cmap="Greys")
plt.title("predicted label: "+str(pred[0]))


