import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(2020)

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

#visualize decision tree
import matplotlib.pyplot as plt
import graphviz
from graphviz import Source
from sklearn import tree

labels = data.feature_names
dot = tree.export_graphviz(clf, feature_names = labels) 
graph = graphviz.Source(dot)
graph.format='png'  
graph.render('dtree',view=True)
