#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 22:48:41 2018

@author: akinyilmaz
"""
import numpy as np


W1 = np.random.randn(16,1,2,2)*0.01
W2 = np.random.randn(32,16,2,2)*0.01
#W3 = np.random.randn(32,16,2,2)*0.01
#W4 = np.random.randn(16,1,2,2)*0.01
W3 = np.random.randn(1568,1568)*0.01

np.savetxt("W1.txt",W1.reshape(1,-1))
np.savetxt("W2.txt",W2.reshape(1,-1))
np.savetxt("W3.txt",W3.reshape(1,-1))
#np.savetxt("W4.txt",W4.reshape(1,-1))
#np.savetxt("W5.txt",W5.reshape(1,-1))

a = np.loadtxt("W5.txt")


train_normal = np.random.normal(90,45,size=(4000,784))
test_normal = np.random.normal(90,45,size=(1000,784))

np.savetxt("train_normal.txt",train_normal.reshape(1,-1))
np.savetxt("test_normal.txt",test_normal.reshape(1,-1))