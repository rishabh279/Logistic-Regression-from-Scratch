# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:37:39 2018

@author: rishabh
Illustrate gradient descent in logistic regression
"""

import numpy as np
import matplotlib.pyplot as plt

N=100
D=2

X=np.random.rand(N,2)

#centered at (-2,-2)
X[:50,:]=X[:50,:]-2*np.ones((50,D))

#centered at (2,2)
X[50:,:]=X[50:,:]+2*np.ones((50,D))

Xb=np.concatenate((np.ones((N,1)),X),axis=1)

w=np.random.rand(D+1).reshape(-1,1)

z=Xb.dot(w)

def sigmoid(z):
  return (1/(1+np.exp(-z)))

Y=sigmoid(z)   
T= np.array([0]*50+[1]*50).reshape(-1,1)
  
def costFunction(T,Y):
  E=0
  for i in range(len(T)):
    if T[i]==1:
     E-=np.log(Y[i]) 
    else:
      E-=np.log(1-Y[i])
      
  return E   
      
learning_rate=0.1
cost=[]
for i in range(100):
  #if i%10==0:
    #print(costFunction(T,Y))
  cost.append(costFunction(T,Y))  
  w=w-learning_rate*Xb.T.dot(Y-T) 
  Y=sigmoid(Xb.dot(w))
  
plt.plot(cost)  

plt.scatter(X[:,0],X[:,1])
  
  