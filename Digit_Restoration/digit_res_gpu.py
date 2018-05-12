# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time


#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def compute_psnr(matris,mse,device):
    max_matris = matris.max()
    psnr = 20*log10((max_matris/(mse**0.5)),device)
    return psnr
    
def log10(x,device):
    numerator = torch.log(x)
    denominator = torch.log(torch.Tensor([10.0]).to(device).float())
    return numerator/denominator

def create_datasets():
    #mean = 90.0
    #std = 45.0
    
    dataset = pd.read_csv('train.csv',sep=',').values
    from sklearn.model_selection import train_test_split
    train_set,test_set = train_test_split(dataset,test_size=0.2,random_state=0)
    
    train_features = train_set[:,1:]
    train_features_noisy = train_features + np.random.randn(33600,784)*0.01
    train_features_noisy = np.clip(train_features_noisy,0.0,255.0)
    
    test_features = test_set[:,1:]
    test_features_noisy = test_features + np.random.randn(8400,784)*0.01
    test_features_noisy = np.clip(test_features_noisy,0.0,255.0)
    
    
    X_train = train_features_noisy.reshape(-1,1,28,28)
    Y_train = train_features.reshape(-1,1,28,28)
    
    X_test = test_features_noisy.reshape(-1,1,28,28)
    Y_test = test_features.reshape(-1,1,28,28)
    
    X_min = np.min(X_train,axis=(0,2,3),keepdims=True)
    X_max = np.max(X_train,axis=(0,2,3),keepdims=True)
    
    X_train = (X_train - X_min) / (X_max-X_min)
    X_test = (X_test - X_min) / (X_max-X_min)
    
    return X_train.astype(np.float64),Y_train.astype(np.float64),X_test.astype(np.float64),Y_test.astype(np.float64)


def initialize_parameters(device):
    W1 = np.loadtxt("W1.txt").reshape(16,1,2,2)
    W2 = np.loadtxt("W2.txt").reshape(32,16,2,2)
    #W3 = np.loadtxt("W3.txt").reshape(32,16,2,2)
    #W4 = np.loadtxt("W4.txt").reshape(16,1,2,2)
    W3 = np.loadtxt("W3.txt").reshape(1568,1568)
    
    parameters = {
            "W1": Variable(torch.from_numpy(W1).to(device).float(),requires_grad=True),
            "W2": Variable(torch.from_numpy(W2).to(device).float(),requires_grad=True),
            "W3": Variable(torch.from_numpy(W3).to(device).float(),requires_grad=True)
            }
    return parameters



def forward(X,parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    
    conv1 = F.conv2d(X,W1,bias=None,padding=0,stride=2)
    relu1 = F.relu(conv1)
    
    conv2 = F.conv2d(relu1,W2,bias=None,padding=0,stride=2)
    relu2 = F.relu(conv2)
    
    f1 = relu2.view(X.size()[0],-1)
    fc = F.relu(torch.mm(f1,W3))
    f2 = fc.view(X.size()[0],32,7,7)
    
    unconv2 = F.conv_transpose2d(f2,W2,bias=None,stride=2,padding=0)
    unrel2 = F.relu(unconv2)

    unconv1 = F.conv_transpose2d(unrel2,W1,bias=None,padding=0,stride=2)
    #out = F.relu(unconv1)
    unconv1 = torch.clamp(unconv1,0.0,255.0)
    
    return unconv1
    
def compute_cost(out,Y):
    cost = torch.mean((out-Y)**2)
    return cost



    
def model(device,X_train,Y_train,X_test,Y_test,num_epoch,learning_rate,batch_size):
    
    epoch_list = []
    cost_list = []
    test_cost_list = []
    train_psnr_list = []
    test_psnr_list = []
    
    parameters = initialize_parameters(device)
    
    optims = [parameters['W1'],parameters['W2'],parameters['W3']]
    
    optimizer = torch.optim.Adam(optims,lr=learning_rate,betas=(0.9,0.999))
    
    total_batch = np.ceil(X_train.shape[0]/batch_size)
    total_test_batch = np.ceil(X_test.shape[0]/batch_size)
    start_time = time.time()
    print("AKi")
    for epoch in range(1,num_epoch+1):     
        total_cost = 0
        total_test_cost = 0
        total_train_psnr = 0
        total_test_psnr = 0
        for batch in range(0,X_train.shape[0],batch_size):
            if X_train.shape[0]-batch < batch_size:
                minibatch_X = torch.from_numpy(X_train[batch:,:,:,:]).to(device).float()
                minibatch_Y = torch.from_numpy(Y_train[batch:,:,:,:]).to(device).float()
            else:
                minibatch_X = torch.from_numpy(X_train[batch:batch+batch_size,:,:,:]).to(device).float()
                minibatch_Y = torch.from_numpy(Y_train[batch:batch+batch_size,:,:,:]).to(device).float()
            
            optimizer.zero_grad()
                
            out = forward(minibatch_X,parameters)
            
            cost = compute_cost(out,minibatch_Y)
            #print(cost.item())
            cost.backward()
            
            optimizer.step()
        
        for batch in range(0,X_train.shape[0],batch_size):
            if X_train.shape[0]-batch < batch_size:
                minibatch_X = torch.from_numpy(X_train[batch:,:,:,:]).to(device).float()
                minibatch_Y = torch.from_numpy(Y_train[batch:,:,:,:]).to(device).float()
            else:
                minibatch_X = torch.from_numpy(X_train[batch:batch+batch_size,:,:,:]).to(device).float()
                minibatch_Y = torch.from_numpy(Y_train[batch:batch+batch_size,:,:,:]).to(device).float()
             
            #print(minibatch_X.type())    
            out = forward(minibatch_X,parameters)
            cost = compute_cost(out,minibatch_Y)
            
            total_cost += cost.item()/total_batch
            
            train_psnr = compute_psnr(out,cost,device)
            total_train_psnr += train_psnr.item()/total_batch
        
        for batch in range(0,X_test.shape[0],batch_size):
            if X_test.shape[0]-batch < batch_size:
                minibatch_test_X = torch.from_numpy(X_test[batch:,:,:,:]).to(device).float()
                minibatch_test_Y = torch.from_numpy(Y_test[batch:,:,:,:]).to(device).float()
            else:
                minibatch_test_X = torch.from_numpy(X_test[batch:batch+batch_size,:,:,:]).to(device).float()
                minibatch_test_Y = torch.from_numpy(Y_test[batch:batch+batch_size,:,:,:]).to(device).float()
            
            test_out = forward(minibatch_test_X,parameters)
            test_cost = compute_cost(test_out,minibatch_test_Y)
            
            total_test_cost += test_cost.item()/total_test_batch
            
            test_psnr = compute_psnr(test_out,test_cost,device)
            total_test_psnr += test_psnr.item()/total_test_batch
        
        train_psnr_list.append(total_train_psnr)
        test_psnr_list.append(total_test_psnr)
        epoch_list.append(epoch)
        cost_list.append(total_cost)
        test_cost_list.append(total_test_cost)
        print("epoch: "+str(epoch))
        print("training cost: "+str(total_cost))
        print("training psnr: "+str(total_train_psnr))
        print("***")
        print("test cost: "+str(total_test_cost))
        print("test psnr: "+str(total_test_psnr))
        print("-----------------")
        
    end_time = time.time()
    return epoch_list,cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_psnr_list,test_psnr_list,end_time-start_time


X_train,Y_train,X_test,Y_test = create_datasets()


device = torch.device("cuda:0")

epoch_list,cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_psnr_list,test_psnr_list,train_time = model(device,
                                                                                                                                 X_train,
                                                                                                                                 Y_train,
                                                                                                                                 X_test,
                                                                                                                                 Y_test,
                                                                                                                                 num_epoch=25,
                                                                                                                                 learning_rate=3.e-2,
                                                                                                                                 batch_size=1024)



plt.plot(epoch_list,cost_list,'-b',label="training cost")
plt.plot(epoch_list,test_cost_list,'-r',label="test cost")
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title("training time: "+str(train_time)+" seconds\n\n"+
          "training cost after 5 iterations: "+str(cost_list[4])+"\n"+
          "training cost after 15 iterations: "+str(cost_list[14])+"\n"+
          "training cost after 25 iterations: "+str(cost_list[24])+"\n")
plt.grid()
plt.legend()

plt.plot(epoch_list,train_psnr_list,'-b',label="training PSNR")
plt.plot(epoch_list,test_psnr_list,'-r',label="test PSNR")
plt.xlabel("# of iterations")
plt.ylabel("PSNR")
plt.title("Final training PSNR: "+str(train_psnr_list[24])+"\n"+
          "Final test PSNR: "+str(test_psnr_list[24]))
plt.grid()
plt.legend()

test_out_n = test_out.data.numpy().astype(np.float64).reshape(-1,28,28,1)

plt.imshow(test_out_n[4,:,:,0],cmap='Greys')
plt.imshow(minibatch_test_X.reshape(-1,28,28,1)[4,:,:,0],cmap='Greys')
plt.imshow(minibatch_test_Y.reshape(-1,28,28,1)[4,:,:,0],cmap='Greys')



plt.imshow(X_test.reshape(-1,28,28,1)[3,:,:,0],cmap='Greys')
plt.imshow(Y_test.reshape(-1,28,28,1)[3,:,:,0],cmap='Greys')