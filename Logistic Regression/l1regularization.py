# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 18:07:04 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt
  
N=50
D=50

#uniformly distributed between (-5,5) 
X=(np.random.random((N,D))-0.5)*10

true_w=np.array([1,0.5,-0.5]+[0]*(D-3))

def sigmoid(z):
  return (1/(1+np.exp(-z)))  

#Adding noise because to spread data [see lasso and ridge article in analytics vidhya] 
Y=np.round(sigmoid(X.dot(true_w)+np.random.randn(N)*0.5))

costs=[]
# w dividing by standard deviation ? DOUBT 
w=np.random.randn(D)/np.sqrt(D)
learning_rate=0.001
l1=3
for i in range(5000):
  Yhat=sigmoid(X.dot(w))
  w=w-learning_rate*(X.T.dot(Yhat-Y)+l1*np.sign(w))
  
  cost=-(Y*np.log(Yhat)+(1-Y)*np.log(1-Yhat)).mean()+l1*np.abs(w).mean()
  costs.append(cost)
  
plt.plot(costs)

plt.plot(true_w,label='true')
plt.plot(w,label='predicted')
plt.legend()
  