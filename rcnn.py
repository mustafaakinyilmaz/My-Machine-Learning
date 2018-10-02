#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import os
from skimage.transform import resize
import cv2
import itertools
import warnings
warnings.filterwarnings('ignore')
import skimage as sk
from natsort import natsorted

device = torch.device("cuda")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


# In[2]:


def create_pair_lists(train_test_path):
    
    class_names = [names for names in glob.glob(train_test_path+"/*")]
    #print(class_names)
    
    a = []
    for names in class_names:
        folder = []
        for folders in glob.glob(names+"/*"):
            folder.append(folders)
        a.append(folder)
        

    paired_folders = itertools.zip_longest(*a)
    
    return list(paired_folders)


# In[3]:


def create_sets(folder,label):
    X_ = []
    Y_ = []
    frame_list = []
    num_of_images = 0
    for frame in glob.glob(folder+"/*.jpg"):
        im_array = im_toarray(frame)
        X_.append(im_array)
        Y_.append(label)
        num_of_images += 1
    
    return normalize(np.array(X_)),np.array(Y_),num_of_images


# In[4]:


def im_toarray(im):
    array = plt.imread(im)
    array = resize(array,(224,224,3),anti_aliasing=True)
    return array


# In[5]:


def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])


# In[6]:


def normalize(X):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    X_normalized = (X - mean)/std
    return X_normalized
    


# In[7]:


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    return tensor


# In[8]:



def model():

    model = models.resnet34(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    p = 0
    for child in model.children():
        if p > 4:
            for param in child.parameters():
                param.requires_grad = True
        p += 1
    
    return model


# In[9]:


class LSTMCell(nn.Module):
    def __init__(self, flatten_dim, hidden_size, bias):
        super(LSTMCell, self).__init__()
    
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size

        self.bias        = bias
        
        self.i2h = nn.Linear(flatten_dim, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        
 
        
    def forward(self, x, cur_state):
        
        if cur_state is None:
            cur_state = (Variable(torch.zeros(1, self.hidden_size)).to(device).float(),
                Variable(torch.zeros(1, self.hidden_size)).to(device).float())
        
        x = x.view(1,-1)
        c_cur, h_cur = cur_state
        preact = self.i2h(x) + self.h2h(h_cur)
        #print(preact.size())
        ingate, forgetgate, cellgate, outgate = preact.chunk(4, 1)
        
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        c_next = (forgetgate * c_cur) + (ingate * cellgate)
        h_next = outgate * F.tanh(c_next)
        
        next_state = (c_next,h_next)
        
        #softmax_out = F.softmax(self.linear(h_next.view(1,-1)),dim=1)
        
        return next_state


# In[10]:


class RESLSTM(nn.Module):
    def __init__(self, original_model):
        super(RESLSTM, self).__init__()
        
        self.cnn = nn.Sequential(*list(original_model.children())[:-1]).to(device).float()
        #self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = LSTMCell(512,512,True).to(device).float()
        self.lstm2 = LSTMCell(512,99,True).to(device).float()
        #self.lstm3 = LSTMCell(128,99,True).to(device).float()

        
    def forward(self,x,state1,state2):
        conv = self.cnn(x)
        c1,h1 = self.lstm1(conv,state1)
        c2,h2 = self.lstm2(h1,state2)
        #c3,h3 = self.lstm3(h2,state3)
        next_state1 = (c1,h1)
        next_state2 = (c2,h2)
        #next_state3 = (c3,h3)
        softmax = F.softmax(h2,dim=1)    
        return next_state1,next_state2,softmax
    


# In[11]:



def calculate_loss(out,real):

    log_loss = torch.mean(-real*torch.log(out))
    mean_loss = torch.mean((out-real)**2)
    
    loss = (mean_loss + log_loss)/2.0

    return log_loss


# In[12]:


def prepare_X_Y(folder_class,target):
    nb_classes = 99
    X,Y,num_of_images = create_sets(folder_class,target)
    X = X.reshape(-1,3,224,224)
    Y = one_hot(Y,nb_classes)
    X,Y = array_to_tensor(X).to(device).float(),array_to_tensor(Y).to(device).float()
    return X,Y,num_of_images


# In[13]:


def train_loader(pair_training,ind):
    train_pairs = list(pair_training[ind])
    notnone = len(train_pairs) - train_pairs.count(None)
    print(str(notnone)+"/"+str(len(train_pairs)))
    target = 0
    
    for folder_class in train_pairs:
        
        if folder_class is not None:
            
            X,Y,num_of_images = prepare_X_Y(folder_class,target)
            yield (folder_class,X,Y,num_of_images,notnone)
            
        target += 1
    


# In[14]:


def main():
    resnet34 = model().to(device).float()
    res_lstm = RESLSTM(resnet34).to(device).float()
    
    pair_training = create_pair_lists(train_test_path= "Train")
    pair_test =  create_pair_lists(train_test_path= "Test")
    
    nb_epoch = 25
    lr = 1.e-1
    optim = filter(lambda p: p.requires_grad,res_lstm.parameters())  
    optimizer = torch.optim.RMSprop(optim,lr)
    
    
    for epoch in range(1,nb_epoch+1):
        epoch_loss = 0
        epoch_correct = 0
        epoch_notnone_videos = 0
        epoch_images = 0
        for ind in range(len(pair_training)):
            train_batch = train_loader(pair_training,ind)
            pair_loss = 0
            pair_correct = 0
            total_pair_images = 0
            for folder_class,X,Y,num_of_images,notnone in train_batch:
                #print(folder_class)
                folder_class_loss,folder_correct = train(X,Y,res_lstm,optimizer)
                pair_loss += folder_class_loss
                pair_correct += folder_correct
                total_pair_images += num_of_images
            
            print("pair: "+str(ind))
            print("there are "+str(total_pair_images)+" images in the pair")
            print("pair loss: "+str(pair_loss.item()/total_pair_images))
            print("pair accuracy: "+str(pair_correct)+"/"+str(notnone))
            print("********")
            epoch_loss += pair_loss
            epoch_correct += pair_correct
            epoch_images += total_pair_images
            epoch_notnone_videos += notnone
        
        print("epoch: "+str(epoch))
        print("there are "+str(epoch_images)+" images in the data")
        print("epoch loss: "+str(epoch_loss.item()/epoch_images))
        print("epoch accuracy: "+str(epoch_correct)+"/"+str(epoch_notnone_videos))
        print("---------------------------")


# In[15]:


def train(X,Y,model,optimizer):
    folder_class_loss = 0
    
    state1 = None
    state2 = None
    softmax_average = 0
    
    for f in range(0,X.size()[0]):
        inp = X[f:f+1]
        real = Y[f:f+1]
                        
        state1,state2,softmax_out = model(inp,state1,state2)
        loss = calculate_loss(softmax_out,real)
        folder_class_loss += loss
        softmax_average += softmax_out
        
    folder_class_average_loss = folder_class_loss/X.size(0)
    softmax_average = softmax_average/X.size(0)
    
    correct = 0
    if torch.argmax(softmax_average,dim=1).item() == torch.argmax(real,dim=1).item():
        correct = 1
    
    optimizer.zero_grad()
                    
    folder_class_average_loss.backward()
                    
    optimizer.step()
    
    return folder_class_loss,correct
    



main()






