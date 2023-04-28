import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt



dane = pd.read_csv(r"C:\Users\julia\Downloads\PSLalphabet.csv")



y = dane["Znak"]
dane.drop("Znak",inplace = True, axis = 1)
x=dane



plt.figure(figsize=(910, 100)) #wielkosc wyswietlanego okna z wykresem
clf = tree.DecisionTreeClassifier(max_depth=(100))
clf.fit(x, y)



tree.plot_tree(clf, fontsize=5) #wielkosc czcionki na wykresach
  