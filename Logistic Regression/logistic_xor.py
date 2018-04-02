# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:38:18 2018

@author: rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

N=4
D=2

#XOR
X=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]])

Y=np.array([0,1,1,0])

ones=np.ones((N,1))
#plt.scatter(X[:,0],X[:,1],c=Y)
xy=(X[:,0]*X[:,1]).reshape(-1,1)
Xb=np.concatenate((ones,xy,X),axis=1)

w=np.random.randn(D+2)


def sigmoid(z):
  return (1/(1+np.exp(-z)))
  
Yhat=sigmoid(Xb.dot(w))  
  
def costFunction(Yhat,Y):
  return -np.mean(Y*np.log(Yhat)+(1-Y)*np.log(1-Yhat))

learning_rate=0.01
costs=[]
for i in range(10000):
  costs.append(costFunction(Yhat,Y))
  w=w-learning_rate*(Xb.T.dot(Yhat-Y)+ 0.01*w)
  Yhat=sigmoid(Xb.dot(w))
plt.plot(costs)

print("Final classification rate:", 1 - np.abs(Y - np.round(Yhat)).sum() / N)
'''
OR  
print((np.round(Yhat)==Y).mean())  
'''
plt.scatter(X[:,0],X[:,1],c=Y)
plt.plot(X,sigmoid(Xb.dot(w)))
plt.legend()
