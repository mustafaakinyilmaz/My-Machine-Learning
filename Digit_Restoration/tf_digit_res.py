#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:58:05 2018

@author: akinyilmaz
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time



def create_placeholders(n_h,n_w,n_c):
    X = tf.placeholder(dtype= tf.float32, shape= [None,n_h,n_w,n_c], name="X")
    Y = tf.placeholder(dtype= tf.float32, shape= [None,n_h,n_w,n_c], name="Y")
    return X,Y

def compute_psnr(matris,mse):
    max_matris = tf.reduce_max(matris)
    #psnr = 20*math.log((max_matris/tf.sqrt(mse)),10)
    psnr = 20*log10(max_matris/tf.sqrt(mse))
    return psnr

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10,dtype=numerator.dtype))   
    return numerator/denominator


def create_datasets():
    #mean = 90.0
    #std = 45.0
    
    dataset = pd.read_csv('train.csv',sep=',').values
    train_set = dataset[5000:9000,:]
    test_set = dataset[2000:3000,:]
    
    train_features = train_set[:,1:]
    train_features_noisy = train_features + np.loadtxt("train_normal.txt").reshape(4000,784)
    train_features_noisy = np.clip(train_features_noisy,0.0,255.0)
    
    test_features = test_set[:,1:]
    test_features_noisy = test_features + np.loadtxt("test_normal.txt").reshape(1000,784)
    test_features_noisy = np.clip(test_features_noisy,0.0,255.0)
    
    
    X_train = train_features_noisy.reshape(-1,28,28,1)
    Y_train = train_features.reshape(-1,28,28,1)
    
    X_test = test_features_noisy.reshape(-1,28,28,1)
    Y_test = test_features.reshape(-1,28,28,1)
    
    X_min = np.min(X_train,axis=(0,1,2),keepdims=True)
    X_max = np.max(X_train,axis=(0,1,2),keepdims=True)
    
    X_train = (X_train - X_min) / (X_max-X_min)
    X_test = (X_test - X_min) / (X_max-X_min)
    
    return X_train.astype(np.float32),Y_train.astype(np.float32),X_test.astype(np.float32),Y_test.astype(np.float32)


def initialize_parameters():
    
    W1 = np.loadtxt("W1.txt").reshape(2,2,1,16)
    W2 = np.loadtxt("W2.txt").reshape(2,2,16,32)
    #W3 = np.loadtxt("W3.txt").reshape(2,2,16,32)
    #W4 = np.loadtxt("W4.txt").reshape(2,2,1,16)
    W3 = np.loadtxt("W3.txt").reshape(1568,1568)
    
    parameters = {
            "W1": tf.Variable(W1,name="W1",trainable=True,dtype=tf.float32),
            "W2": tf.Variable(W2,name="W2",trainable=True,dtype=tf.float32),
            "W3": tf.Variable(W3,name="W3",trainable=True,dtype=tf.float32),
            }
    return parameters


def forward(X,parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    
    conv1 = tf.nn.conv2d(X,W1,strides=[1,2,2,1],padding="VALID")
    relu1 = tf.nn.relu(conv1)
    
    conv2 = tf.nn.conv2d(relu1,W2,strides=[1,2,2,1],padding="VALID")
    relu2 = tf.nn.relu(conv2)
    
    f1 = tf.reshape(relu2,shape=[tf.shape(X)[0],-1])
    fc = tf.nn.relu(tf.matmul(f1,W3))
    f2 = tf.reshape(fc,shape=[tf.shape(X)[0],7,7,32])
    
    unconv2 =  tf.nn.conv2d_transpose(f2,W2,strides=[1,2,2,1],output_shape=[tf.shape(X)[0],14,14,16],padding="VALID")
    unrel2 = tf.nn.relu(unconv2)
    
    unconv1 = tf.nn.conv2d_transpose(unrel2,W1,strides=[1,2,2,1],output_shape=[tf.shape(X)[0],28,28,1],padding="VALID")
    unconv1 = tf.clip_by_value(unconv1,0.0,255.0)
    
    return unconv1

def compute_cost(out,Y):
    cost = tf.reduce_mean(tf.pow(tf.subtract(out,Y),2))
    return cost
    

def model(X_train,Y_train,X_test,Y_test,num_epoch,learning_rate,batch_size):
    tf.reset_default_graph()
    (m,n_h,n_w,n_c) = X_train.shape
    
    epoch_list = []
    train_cost_list = []
    test_cost_list = []
    train_psnr_list = []
    test_psnr_list = []
    
    total_batch = np.ceil(X_train.shape[0]/batch_size)  
    total_test_batch = np.ceil(X_test.shape[0]/batch_size)
    
    X,Y = create_placeholders(n_h,n_w,n_c)
    
    parameters = initialize_parameters()
    
    out = forward(X,parameters)
    cost = compute_cost(out,Y)
    psnr = compute_psnr(out,cost)
    
    optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.9,beta2=0.99,epsilon=1.e-8).minimize(cost,var_list=parameters)
    
    init = tf.global_variables_initializer()
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1,num_epoch+1):   
            total_cost = 0
            total_test_cost = 0
            total_train_psnr = 0
            total_test_psnr = 0
            for batch in range(0,X_train.shape[0],batch_size):
                if X_train.shape[0]-batch < batch_size:
                    minibatch_X = X_train[batch:,:,:,:]
                    minibatch_Y = Y_train[batch:,:,:,:]
                else:
                    minibatch_X = X_train[batch:batch+batch_size,:,:,:]
                    minibatch_Y = Y_train[batch:batch+batch_size,:,:,:]
              
                sess.run(optimizer,feed_dict= {X: minibatch_X, Y: minibatch_Y})
            
            for batch in range(0,X_train.shape[0],batch_size):
                if X_train.shape[0]-batch < batch_size:
                    minibatch_X = X_train[batch:,:,:,:]
                    minibatch_Y = Y_train[batch:,:,:,:]
                else:
                    minibatch_X = X_train[batch:batch+batch_size,:,:,:]
                    minibatch_Y = Y_train[batch:batch+batch_size,:,:,:]
                
                batch_cost = sess.run(cost,feed_dict={X:minibatch_X,Y:minibatch_Y})
                total_cost += batch_cost/total_batch
                
                train_psnr = sess.run(psnr,feed_dict={X:minibatch_X,Y:minibatch_Y})
                total_train_psnr += train_psnr/total_batch
            
            for batch in range(0,X_test.shape[0],batch_size):
                if X_test.shape[0]-batch < batch_size:
                    minibatch_test_X = X_test[batch:,:,:,:]
                    minibatch_test_Y = Y_test[batch:,:,:,:]
                else:
                    minibatch_test_X = X_test[batch:batch+batch_size,:,:,:]
                    minibatch_test_Y = Y_test[batch:batch+batch_size,:,:,:]
                    
                test_out,test_batch_cost = sess.run([out,cost],feed_dict={X:minibatch_test_X,Y:minibatch_test_Y})
                total_test_cost += test_batch_cost/total_test_batch
                
                test_psnr = sess.run(psnr,feed_dict={X:minibatch_test_X,Y:minibatch_test_Y})
                total_test_psnr += test_psnr/total_test_batch
                
            train_psnr_list.append(total_train_psnr)
            test_psnr_list.append(total_test_psnr)
            epoch_list.append(epoch)
            train_cost_list.append(total_cost)
            test_cost_list.append(total_test_cost)
            print("epoch: "+str(epoch))
            print("training cost: "+str(total_cost))
            print("training psnr: "+str(total_train_psnr))
            print("***")
            print("test cost: "+str(total_test_cost))
            print("test psnr: "+str(total_test_psnr))
            print("-----------------")
        
        end_time = time.time()

    return epoch_list,train_cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_psnr_list,test_psnr_list,end_time-start_time

            
            
                

X_train,Y_train,X_test,Y_test = create_datasets()


epoch_list,train_cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_psnr_list,test_psnr_list,train_time = model(X_train,
                                                                                                                                       Y_train,
                                                                                                                                       X_test,
                                                                                                                                       Y_test,        
                                                                                                                                       num_epoch=25,
                                                                                                                                       learning_rate=3.e-2,
                                                                                                                                       batch_size=64)


plt.plot(epoch_list,train_cost_list,'-b',label="training cost")
plt.plot(epoch_list,test_cost_list,'-r',label="test cost")
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title("training time: "+str(train_time)+" seconds\n\n"+
          "training cost after 5 iterations: "+str(train_cost_list[4])+"\n"+
          "training cost after 15 iterations: "+str(train_cost_list[14]) +"\n"+
          "training cost after 25 iterations: "+str(train_cost_list[24]))
plt.legend()
plt.grid()



plt.plot(epoch_list,train_psnr_list,'-b',label="training PSNR")
plt.plot(epoch_list,test_psnr_list,'-r',label="test PSNR")
plt.xlabel("# of iterations")
plt.ylabel("PSNR")
plt.title("Final training PSNR: "+str(train_psnr_list[24])+"\n"+
          "Final test PSNR: "+str(test_psnr_list[24]))
plt.grid()
plt.legend()

plt.imshow(test_out[4,:,:,0],cmap='Greys')
plt.imshow(minibatch_test_X[3,:,:,0],cmap='Greys')
plt.imshow(minibatch_test_Y[3,:,:,0],cmap='Greys')










X_train,Y_train,X_test,Y_test = create_datasets()
    




X_batch = X_train[0:100,:,:,:]
Y_batch = Y_train[0:100,:,:,:]

    

(m,n_h,n_w,n_c) = X_batch.shape

graph = tf.Graph()

with graph.as_default():
    
    
    """X,Y = create_placeholders(n_h,n_w,n_c)
    parameters = initialize_parameters()

    out = forward(X,parameters)
    cost = compute_cost(out,Y)"""
    W1 = 2*np.random.randn(3,3)
    W1 = tf.Variable(W1,name="W1",trainable=True,dtype=tf.float32)
    clipped = tf.clip_by_value(W1,0,2)
    
    init = tf.global_variables_initializer()
    
with tf.Session(graph=graph) as sess:
    sess.run(init)
    
    a = W1.eval()
    c = clipped.eval()
    
    """c = sess.run(cost,feed_dict={X:X_batch,Y:Y_batch})
    o = sess.run(out,feed_dict={X:X_batch,Y:Y_batch})"""

    