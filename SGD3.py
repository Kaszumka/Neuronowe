# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:08:05 2022

@author: huber
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

import pandas as pd
c=pd.read_csv('C:/Users/huber/OneDrive/Pulpit/PSLalphabet.csv')
y=c['Znak']  
znaki=['P2_1','P3_1']
P=c[znaki]
c.drop('Znak',inplace=True, axis=1)
clf = SGDClassifier(loss="hinge", alpha=0.01)
clf.fit(P, y)
for i in P:
    
    R=clf.predict(P)
print(R)

X=P.values

print(P)
print(R)

#plt.plot(X,R)
#for i in X,Y:
    
#    plt.scatter(X, Y, c='y',  edgecolor="black", s=20)

#plt.axis("tight")
#plt.show()
trening=int(0.6*len(X))
P_train,P_test=X[:trening],X[trening:]
y_train,y_test=y[:trening],y[trening:]
plt.scatter(P_train[:,0],P_train[:,1],c='g',edgecolors="black")
plt.scatter(P_test[:, 0], P_test[:, 1], c='r' ,alpha=0.6,edgecolors="yellow")