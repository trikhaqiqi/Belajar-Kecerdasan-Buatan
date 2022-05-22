# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 20:12:26 2022

@author: trikhaqiqi
"""
"""
    Vektorisasi data
"""

print("-----Ini vektorisasi data-----")

#
import pandas as pd
d = pd.read_csv("MOCK_DATA.csv")

awalan = d.head()

d.shape 

print(awalan)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dvec = vectorizer.fit_transform(d['stock_name'])
print(dvec)

daptarkata = vectorizer.get_feature_names()
print(daptarkata)

dshuf = d.sample(frac=1)

d_train = dshuf[:450]
d_test = dshuf[450:]

d_train_att = vectorizer.fit_transform(d_train['stock_name'])
print(d_train_att)

d_test_att = vectorizer.transform(d_test['stock_name'])
print(d_test_att)

d_train_label = d_train['stock_market']
print(d_train_label)
d_test_label = d_test['stock_market']
print(d_test_label)


#%%
"""
    Praktek
"""

"""
    Vektorisasi data
"""

print("-----Ini vektorisasi data-----")

import pandas as pd
d = pd.read_csv("Youtube04-Eminem.csv")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dvec = vectorizer.fit_transform(d['CONTENT'])
print(dvec)

daptarkata = vectorizer.get_feature_names()
print(daptarkata)

#%%

dshuf = d.sample(frac=1)

d_train = dshuf[:300]
d_test = dshuf[300:]

d_train_att = vectorizer.fit_transform(d_train['CONTENT'])
print(d_train_att)

d_test_att = vectorizer.transform(d_test['CONTENT'])
print(d_test_att)

d_train_label = d_train['CLASS']
print(d_train_label)

d_test_label = d_test['CLASS']
print(d_test_label)

#%%

"""
    Mencoba dengan metode Decission Tree dan SVM
"""

"""
    Menggunakaan Dession tree
"""
print("-----Menggunakan dession tree")

from sklearn import tree

clftree = tree.DecisionTreeClassifier()
clftree.fit(d_train_att, d_train_label)
clftree.score(d_test_att, d_test_label)
print('===ini menggunakan decision tree')
print(clftree.score(d_test_att, d_test_label))

"""
    Menggunakan SVM
"""
print("-----Menggunakan SVM")

from sklearn import svm

clfsvm = svm.SVC()
clfsvm.fit(d_train_att, d_train_label)
clfsvm.score(d_test_att, d_test_label)
print('===ini menggunakan svm')
print(clfsvm.score(d_test_att, d_test_label))


#%%

"""
    Confusion Matrix
"""

print("-----Ini confusion matrix-----")

from sklearn.metrics import confusion_matrix

pred_labels = clftree.predict(d_test_att)
cm = confusion_matrix(d_test_label, pred_labels)

print(cm)

# Pengecekkan Cross Validation
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(clftree, d_train_att, d_train_label, cv=5)
# show average score and +/- two standard deviations away ( covering 95% of scores )
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# untuk decission tree
scorestree = cross_val_score(clftree, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

# untuk SVM
scoressvm = cross_val_score(clfsvm, d_train_att, d_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# pengamatan komponen informasi
max_features_opts = range(1, 10, 1)
n_estimators_opts = range(2, 40, 4)
rf_params = np.empty((len(max_features_opts) * len(n_estimators_opts), 4), float)
i = 0

for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accurancy: %0.2f(+/- %0.2f)" % (max_features, n_estimators, scores.mean(), scores.std() * 2))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
fig.clf()

ax = fig.gca(projection='3d')
x = rf_params[:, 0]
y = rf_params[:, 1]
z = rf_params[:, 2]
ax.scatter(x, y, z)
ax.set_zlim(0.6, 1)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()
print(plt.show())

#%%
"""
    Vektorisasi data
"""

print("-----Ini vektorisasi data-----")

import pandas as pd
d = pd.read_csv("Youtube01-Psy.csv")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

dvec = vectorizer.fit_transform(d['CONTENT'])
print(dvec)

daptarkata = vectorizer.get_feature_names()
print(daptarkata)

dshuf = d.sample(frac=1)

d_train = dshuf[:300]
d_test = dshuf[300:]

d_train_att = vectorizer.fit_transform(d_train['CONTENT'])
print(d_train_att)

d_test_att = vectorizer.transform(d_test['CONTENT'])
print(d_test_att)

d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']


# Cari svm

"""
    Klasifikasi dengan Random Forest
"""

print("-----Ini klasifikasi dengan menggunakan random forest")

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=80)

clf.fit(d_train_att, d_train_label)

clf.predict(d_test_att)
print(clf.predict(d_test_att))

clf.score(d_test_att, d_test_label)
print(clf.score(d_test_att, d_test_label))

"""
    Confusion Matrix
"""

print("-----Ini confusion matrix-----")

from sklearn.metrics import confusion_matrix

pred_labels = clf.predict(d_test_att)
cm = confusion_matrix(d_test_label, pred_labels)

print(cm)

"""
    Pengecekan cross validation
"""
print("-----Ini pengecekan cross validation-----")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)

skorrata2 = scores.mean()
print(skorrata2)

skoresd = scores.std()
print(skoresd)
#%%























