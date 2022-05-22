# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 09:44:10 2022
@author: trikhaqiqi Kudang Koding
"""


import pandas as pd
durian = pd.read_csv('student-mat.csv', sep=';')
len(durian)
print(len(durian))


durian['pass'] = durian.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
durian = durian.drop(['G1', 'G2', 'G3'], axis=1)
durian.head()
print(durian.head())

durian = pd.get_dummies(durian, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])

durian.head()
print(durian.head())



durian = durian.sample(frac=1)

durian_train = durian[:300]
durian_test = durian[300:]


durian_train_att = durian_train.drop(['pass'], axis=1)
durian_train_pass = durian_train['pass']

durian_test_att = durian_test.drop(['pass'], axis=1)
durian_test_pass = durian_test['pass']

durian_att = durian.drop(['pass'], axis=1)
durian_pass = durian['pass']



import numpy as np 
print("Passing: %d out of %d (%.2f%%)" % (np.sum(durian_pass), len(durian_pass), 100*float(np.sum(durian_pass)) / len(durian_pass)))


from sklearn import tree 
timun = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
timun = timun.fit(durian_train_att, durian_train_pass)

print(timun)

import graphviz

delima_data = tree.export_graphviz(timun, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(durian_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True)

graph = graphviz.Source(delima_data)
print(graph)


tree.export_graphviz(timun, out_file='student-performance.dot', label="all", impurity=False, proportion=True,
                     feature_names=list(durian_train_att), class_names=['fail', 'pass'],
                     filled=True, rounded=True)

timun.score(durian_test_att, durian_test_pass)
print(timun.score(durian_test_att, durian_test_pass))



from sklearn.model_selection import cross_val_score
salak = cross_val_score(timun, durian_att, durian_pass, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (salak.mean(), salak.std() * 2))


for max_depth in range(1, 20):
    timun = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(timun, durian_att, durian_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, salak.mean(), salak.std() * 2))


duku = np.empty((19, 3), float)
ilwara = 0

for max_depth in range(1, 20):
    timun = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    salak = cross_val_score(timun, durian_att, durian_pass, cv=5)
    duku[ilwara,0] = max_depth
    duku[ilwara,1] = salak.mean()
    duku[ilwara,2] = salak.std() * 2
    ilwara += 1
    
print(duku)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(duku[:,0], duku[:,1], yerr=duku[:,2])
plt.show()


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(duku[:,0], duku[:,1], yerr=duku[:,2])
plt.show()
