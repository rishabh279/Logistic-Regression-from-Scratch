# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 02:36:04 2018

@author: rishabh
L2 regularization here cost function may not be dcreasing at end becoz of random data
"""

import numpy as np
import matplotlib.pyplot as plt

N=100
D=2

X=np.random.rand(N,2)

#data centered at (-2,-2)
X[:50,:]=X[:50,:]-2*np.ones((50,D))

#data centered at (2,2)
X[50:,:]=X[50:,:]+2*np.ones((50,D))

Xb=np.concatenate((np.ones((N,1)),X),axis=1)


Y=np.array([0]*50+[1]*50)

w=np.random.rand(D+1)

def sigmoid(z):
  return (1/(1+np.exp(-z)))
  
  
def costFunction(Yhat,Y):
  return -np.mean(Y*np.log(Yhat)+((1-Y)*np.log(1-Yhat)))  
  
learning_rate=0.1
cost=[]
for i in range(100):
  Yhat=sigmoid(Xb.dot(w))
  w=w-learning_rate*(Xb.T.dot(Yhat-Y)+0.1*w)
  cost_learning=costFunction(Yhat,Y)
  cost.append(cost_learning)

    

  
  
