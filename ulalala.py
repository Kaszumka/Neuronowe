# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 22:21:31 2022

@author: huber
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
h = 0.02
c=pd.read_csv('C:/Users/huber/OneDrive/Pulpit/PSLalphabet.csv')
y =c['Znak']  
c.drop('Znak',inplace=True, axis=1)
P=c.values
clf = MLPClassifier()


clf.fit(P, y)

trening=int(0.6*len(P))

P_train,P_test=P[:trening],P[trening:]
y_train,y_test=y[:trening],y[trening:]
clf = MLPClassifier()
clf.fit(P_train,y_train)
pred_train=clf.predict(P_train)
pred_test=clf.predict(P_test)
acc_train=accuracy_score(y_train,pred_train)
acc_test=accuracy_score(y_test,pred_test)
print('Accuracy:{:.4%}'.format(acc_train))

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
plt.scatter(P_train[:,0],P_train[:,1],c='g',cmap=cm_bright,edgecolors="black")
plt.scatter(P_test[:, 0], P_test[:, 1], c='r', cmap=cm_bright, alpha=0.6,edgecolors="yellow")

G=P_test[:,0]

