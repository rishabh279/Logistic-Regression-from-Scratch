# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:53:06 2018

@author: rishabh
Illustrate cost function in logistic regression
"""

import numpy as np
import matplotlib.pyplot as plt
 
N=100
D=2

X=np.random.rand(N,D)

#data centered at (-2,-2)
X[:50,:]=X[:50,:]-2*np.ones((50,D))

#data centered at (2,2)
X[50:,:]=X[50:,:]+2*np.ones((50,D))

#half 0 and half 1 
T=np.array([0]*50+[1]*50)

ones= np.ones((N,1))
Xb=np.concatenate((ones,X),axis=1)

w=np.random.rand(D+1).reshape(-1,1)

z=Xb.dot(w)

def sigmoid(Z):
  return (1/(1+np.exp(-z)))
  
def costFunction(T,Y):
  E=0  
  for i in range(len(T)):
    if T[i]==1:
      E-=np.log(Y[i])
    else:
      E-=np.log(1-Y[i])
  return E
      
print(costFunction(T,sigmoid(z)))

#closed form solution
'''
w=np.array([0,4,4]).reshape(-1,1)
z=Xb.dot(w)
print(costFunction(T,sigmoid(z)))

#y=-x

plt.scatter(X[:,0],X[:,1],c=T,s=100,alpha=1)
x_axis=np.linspace(-6,6,100)
y_axis=-x_axis
plt.plot(x_axis,y_axis)
plt.show()
'''


