#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:07:14 2018

@author: akinyilmaz
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time



def create_placeholders(n_w,n_h,n_c,n_y):
    X = tf.placeholder(dtype= tf.float32, shape= [None,n_w,n_h,n_c])
    Y = tf.placeholder(dtype= tf.float32, shape= [None,n_y])
    return X,Y



def initialize_parameters():
    
    W1 = tf.Variable(np.loadtxt("W1.txt").reshape(3,3,1,32),name="W1",trainable=True,dtype=tf.float32)
    W2 = tf.Variable(np.loadtxt("W2.txt").reshape(3,3,32,64),name="W2",trainable=True,dtype=tf.float32)
    W3 = tf.Variable(np.loadtxt("W3.txt").reshape(576,10),name="W3",trainable=True,dtype=tf.float32)
    
    
    
    """W1 = tf.Variable(np.random.randn(3,3,1,32)*0.01,name="W1",trainable=True,dtype=tf.float32)
    W2 = tf.Variable(np.random.randn(3,3,32,64)*0.01,name="W2",trainable=True,dtype=tf.float32)
    W3 = tf.Variable(np.random.randn(fc_size,nb_class)*0.01,name="W3",trainable=True,dtype=tf.float32)"""
    
    parameters = {
            "W1": W1,
            "W2": W2,
            "W3": W3}
    
    return parameters



def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    
    Z1 = tf.nn.conv2d(X,W1,strides=[1,3,3,1],padding="VALID")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,1,1,1],strides=[1,1,1,1],padding="VALID")
    
    W2 = parameters["W2"]
    
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,3,3,1],padding="VALID")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,1,1,1],strides=[1,1,1,1],padding="VALID")
    
    F = tf.contrib.layers.flatten(P2)
    
    W3 = parameters["W3"]
    
    Z3 = tf.nn.softmax(tf.matmul(F,W3),axis=1)
    
    #Z3 = tf.contrib.layers.fully_connected(F, 10, activation_fn=None)
    return Z3



def compute_cost(Z3,Y):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Z3),reduction_indices=[1]))
    return cost


    
def model(X_train, Y_train, X_test, Y_test, num_epoch= 10, batch_size= 512, learning_rate0= 1.e-3):
    tf.reset_default_graph()
    (m,n_h,n_w,n_c) = X_train.shape
    n_y = Y_train.shape[1]
    
    costs = []
    epoch_list = []
    train_accur_list = []
    test_accur_list = []

    
    X,Y = create_placeholders(n_w,n_h,n_c,n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate0,beta1=0.9,beta2=0.999,epsilon=1.e-8).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        sess.run(init)
        
        
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
                
                sess.run(optimizer,feed_dict= {X: minibatch_X, Y: minibatch_Y})
                
                batch_cost = sess.run(cost,feed_dict= {X: minibatch_X, Y: minibatch_Y})
                total_cost += batch_cost/total_batch
                
            
            y_pred_train,y_true_train = sess.run([tf.argmax(Z3,1),tf.argmax(Y,1)],
                                     feed_dict= {X: X_train, Y: Y_train})
            
            y_pred_test,y_true_test = sess.run([tf.argmax(Z3,1),tf.argmax(Y,1)],
                                     feed_dict= {X: X_test, Y: Y_test})
            
            
            correct_train = 0
            correct_test = 0
            for k in range(len(y_pred_train)):
                if y_pred_train[k] == y_true_train[k]:
                    correct_train += 1
            
            for k in range(len(y_pred_test)):
                if y_pred_test[k] == y_true_test[k]:
                    correct_test += 1
            
            #total_cost = sess.run(cost,feed_dict= {X:X_train, Y:Y_train})
            train_accuracy = correct_train/len(y_pred_train)
            test_accuracy = correct_test/len(y_pred_test)
            
            
            print("epoch: "+str(epoch))
            print("cost: "+str(total_cost))
            print("train accuracy: "+str(train_accuracy*100))
            print("test accuracy: "+str(test_accuracy*100))
            costs.append(total_cost)
            epoch_list.append(epoch)
            train_accur_list.append(train_accuracy*100)
            test_accur_list.append(test_accuracy*100)
            print("-------------------------------")
        
        return train_accur_list,test_accur_list,costs,epoch_list,y_pred_test,y_true_test



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
    
    X_train = train_features.reshape(-1,28,28,1)
    Y_train = train_results
    
    X_test = test_features.reshape(-1,28,28,1)
    Y_test = test_results
    
    return X_train,Y_train,X_test,Y_test






X_train,Y_train,X_test,Y_test = create_datasets()


start_time = time.time()
train_accurs,test_accurs,costs,epochs,y_pred_test,y_true_test = model(X_train,Y_train,X_test,Y_test)
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
plt.show()



