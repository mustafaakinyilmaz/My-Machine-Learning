#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:03:26 2018

@author: akinyilmaz
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt


def create_datasets():
    dataset = pd.read_csv('train.csv',sep=',').values

    """from sklearn.model_selection import train_test_split
    train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=0)"""
    
    train_set = dataset[:-9000,:]
    test_set = dataset[-9000:,:]
    
    train_features = train_set[:,1:]
    train_results = train_set[:,0].reshape(-1,1)

    test_features = test_set[:,1:]
    test_results = test_set[:,0].reshape(-1,1)

    train_features = train_features/255
    test_features = test_features/255
    
    from sklearn.preprocessing import OneHotEncoder
    oneEn = OneHotEncoder(categorical_features=[0])
    oneEn2 = OneHotEncoder(categorical_features=[0])

    train_results = oneEn.fit_transform(train_results).toarray()
    test_results = oneEn2.fit_transform(test_results).toarray()
    
    X_train = train_features.reshape(-1,1,28,28)
    Y_train = train_results
    
    X_test = test_features.reshape(-1,1,28,28)
    Y_test = test_results
    
    return X_train,Y_train,X_test,Y_test


def initialize_parameters():
    """W1 = np.random.randn(32,1,3,3)*0.01
    W2 = np.random.randn(64,32,3,3)*0.01
    W3 = np.random.randn(576,10)*0.01"""
    
    W1 = np.loadtxt("W1.txt").reshape(32,1,3,3)
    W2 = np.loadtxt("W2.txt").reshape(64,32,3,3)
    W3 = np.loadtxt("W3.txt").reshape(576,10)
    
    parameters = {
            "W1": Variable(torch.from_numpy(W1).float(),requires_grad=True),
            "W2": Variable(torch.from_numpy(W2).float(),requires_grad=True),
            "W3": Variable(torch.from_numpy(W3).float(),requires_grad=True)}
    return parameters


"""W1 = np.random.randn(32,1,3,3)*0.01
W2 = np.random.randn(64,32,3,3)*0.01
W3 = np.random.randn(576,10)*0.01

np.savetxt("W1.txt",W1.reshape(1,-1))
np.savetxt("W2.txt",W2.reshape(1,-1))
np.savetxt("W3.txt",W3.reshape(1,-1))"""


def forward(X,parameters):
    
    (m,n_h,n_w,n_c) = X.shape
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    X = Variable(torch.from_numpy(X).float(),requires_grad=False)
    
    conv1 = F.conv2d(X,W1,bias=None,padding=0,stride=(3,3))
    relu1 = F.relu(conv1)
    pool1 = F.max_pool2d(relu1,kernel_size=(1,1),stride=(1,1),padding=0)
    
    conv2 = F.conv2d(pool1,W2,bias=None,padding=0,stride=(3,3))
    relu2 = F.relu(conv2)
    pool2 = F.max_pool2d(relu2,kernel_size=(1,1),stride=(1,1),padding=0)
    
    
    flatten = pool2.view(m,-1)
    #print(flatten)
    #out = torch.nn.Linear(flatten.size()[1],10)(flatten)
    #out = F.softmax(torch.mm(flatten,W3),dim=0)
    
    out = F.softmax(torch.mm(flatten,W3),dim=1)
    
    return out


def compute_cost(out,Y):
    Y = Variable(torch.from_numpy(Y).float(),requires_grad=False)
    cost = torch.mean(torch.sum(-Y*torch.log(out),dim=1))
    return cost


"""X_train,Y_train,X_test,Y_test = create_datasets()

parameters = initialize_parameters()


cost = compute_cost(out,Y_train[0:5,:])"""





def model(X_train,Y_train,X_test,Y_test,num_epoch=10,batch_size=512,learning_rate=1.e-3):
    (m,nc,n_h,n_w) = X_train.shape
    
    costs = []
    epoch_list = []
    train_accur_list = []
    test_accur_list = []
    

    parameters = initialize_parameters()

    optims = [parameters['W1'],parameters['W2'],parameters['W3']]
    
    optimizer = torch.optim.Adam(optims,lr=learning_rate,betas=(0.9,0.999))
    
    for epoch in range(1,num_epoch+1):     
        total_cost = 0
        total_batch = np.ceil(X_train.shape[0]/batch_size)
        for batch in range(0,X_train.shape[0],batch_size):
            if X_train.shape[0]-batch < batch_size:
                minibatch_X = X_train[batch:,:,:,:]
                minibatch_Y = Y_train[batch:,:]
            else:
                minibatch_X = X_train[batch:batch+batch_size,:,:,:]
                minibatch_Y = Y_train[batch:batch+batch_size,:]
            
            
            optimizer.zero_grad()

            out = forward(minibatch_X,parameters)
            cost = compute_cost(out,minibatch_Y)
            
            total_cost += cost.data.numpy()[0]/total_batch
            
            cost.backward()
            
            optimizer.step()

        
        #total_cost = compute_cost(forward(X_train,parameters),Y_train)
        
        y_pred_train = np.argmax(forward(X_train,parameters).data.numpy(),axis=1)
        y_true_train = np.argmax(Y_train,axis=1)
        
        y_pred_test = np.argmax(forward(X_test,parameters).data.numpy(),axis=1)
        y_true_test = np.argmax(Y_test,axis=1)
        
        
        correct_train = 0
        correct_test = 0
        
        for l in range(len(y_pred_train)):
            if y_pred_train[l] == y_true_train[l]:
                correct_train += 1
        
        for l in range(len(y_pred_test)):
            if y_pred_test[l] == y_true_test[l]:
                correct_test += 1
                
                

        train_accuracy = (correct_train/len(y_pred_train))*100
        test_accuracy = (correct_test/len(y_pred_test) )*100   
        
        costs.append(total_cost)
        epoch_list.append(epoch)
        train_accur_list.append(train_accuracy)
        test_accur_list.append(test_accuracy)
        print("epoch: "+str(epoch))
        print("cost: "+str(total_cost))
        print("train accuracy: "+str(train_accuracy))
        print("test accuracy: "+str(test_accuracy))
        print("-------------")
    
    
    return epoch_list,costs,train_accur_list,test_accur_list,y_pred_test,y_true_test
        
        
        


X_train,Y_train,X_test,Y_test = create_datasets()

start_time = time.time()
epochs,costs,train_accurs,test_accurs,y_pred_test,y_true_test=model(X_train,Y_train,X_test,Y_test)
end_time = time.time()



plt.plot(epochs,train_accurs,'-b',label="train accuracy")
plt.plot(epochs,test_accurs,'-r',label="test accuracy")
plt.xlabel("# of iterations")
plt.ylabel("% accuracy")
plt.title("Training time: "+str(end_time-start_time)+"sec")
plt.legend()
plt.grid()
plt.show()

plt.plot(epochs,costs,'-b',label="cost")
plt.xlabel("# of iterations")
plt.ylabel("cost")
plt.legend()
plt.grid()
