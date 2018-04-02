# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:20:13 2018

@author: rishabh
Illustrate sigmoid function in logistic regression
"""

import numpy as np

N=100
D=2

X=np.random.randn(N,D)
ones=np.ones((N,1))
#Xb=np.concatenate((ones,X),axis=1)

w=np.random.rand(D,1)
z=X.dot(w)
z=np.add(z,ones)

def sigmoid(z):
  return 1/(1+np.exp(-z))

print(sigmoid(z))