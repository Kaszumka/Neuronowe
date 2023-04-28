# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:12:55 2022

@author: huber
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
#METODA NEURONOWA
h = 0.02
c=pd.read_csv('C:/Users/huber/OneDrive/Pulpit/PSLalphabet.csv')
y=c['Znak']  
znaki=['P2_1','P3_1']
P=c[znaki]
c.drop('Znak',inplace=True, axis=1)
clf = MLPClassifier()
clf.fit(P, y)
for i in P:
    
    R=clf.predict(P)
    

P=P.values

trening=int(0.6*len(P))
P_train,P_test=P[:trening],P[trening:]
y_train,y_test=y[:trening],y[trening:]
clf = MLPClassifier()
clf.fit(P_train,y_train)
pred_train=clf.predict(P_train)
pred_test=clf.predict(P_test)
acc_train=accuracy_score(y_train,pred_train)
acc_test=accuracy_score(y_test,pred_test)
print('Prawdopodobieństwo poprawnego wyniku:{:.4%}'.format(acc_train))

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

#G=P_train[:,0]

#XX,YY=np.meshgrid(G,y_train)
#plt.contourf(XX, YY, c, cmap=cm, alpha=0.8)
#R=clf.predict([[26663,10633,16293,21143,17003,20663,19603,15753,14234,18683]])
print(R)
plt.plot(P,R)