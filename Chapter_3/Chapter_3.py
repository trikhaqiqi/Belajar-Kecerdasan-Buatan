# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:19:00 2022

@author: trikh
"""

# membuat aplikasi dengan menggunakan pandas

import pandas as pd

data = {
    'apel': [3, 2, 0, 1], 
    'pisang': [0, 3, 7, 2]
}

beli = pd.DataFrame(data)

print(beli)

beli = pd.DataFrame(data, index=['Ucup', 'Otong', 'Mimin', 'David'])

print(beli)

print('=====Hasil=====')

beli.loc['Ucup']

print(beli.loc['Ucup'])

#%%
# membuat aplikasi dengan menggunakan numpy


import numpy as np

print(np.__version__)

a = np.array([[1,2,3],
              [4,5,6]])

b = np.array([[10,11,12],
              [13,14,15]])

c = a + b

print('===== hasil a + b =====')

print(c)

a = np.array([[1,2,3],
              [4,5,6]])

b = 2*a # multiplying the numpy array a(matrix) by 2

print('===== hasil 2 * a =====')

print(b)

print('===== membaca data dengan numpy')

# Importing libraries that will be used
import numpy as np
 
# Setting name of the file that the data is to be extracted from in python
filename = 'example1.txt'
 
# Loading file data into numpy array and storing it in variable called data_collected
data_collected = np.loadtxt(filename)
 
# Printing data stored
print(data_collected)
 
 
# Type of data
print(
    f'Stored in : {type(data_collected)} and data type is : {data_collected.dtype}')

print('===== membaca data dengan numpy ke 2')


# Importing libraries that will be used
import numpy as np
 
# Setting name of the file that the data is to be extracted from in python
# This is a comma separated values file
filename = 'example1.csv'
 
# Loading file data into numpy array and storing it in variable.
# We use a delimiter that basically tells the code that at every ',' we encounter,
# we need to treat it as a new data point.
# The data type of the variables is set to be int using dtype parameter.
data_collected = np.loadtxt(filename, delimiter=',', dtype=int)
 
# Printing data stored
print(data_collected)
 
 
# Type of data
print(
    f'Stored in : {type(data_collected)} and data type is : {data_collected.dtype}')

#%%
# membuat aplikasi sederhana dengan menggunakan matplotlib
from matplotlib import pyplot as plt
 
plt.bar([0.25,1.25,2.25,3.25,4.25],[50,40,70,80,20], label="BMW",color='y', width=.5)
plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60], label="Mercy", color='g', width=.5)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Distance (kms)')
plt.title('Information')
plt.show()


#%%
# Random Forest

print('=====Random Forest=====')

import pandas as pd 

imgatt = pd.read_csv("data/CUB_200_2011/attributes/image_attribute_labels.txt", sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False, usecols=[0,1,2], names=['imgid', 'attid', 'present'])

print('=====contoh 1')

imgatt.head()
print(imgatt.head())

imgatt.shape
print(imgatt.shape)



# cek isi menggunakan perintah listing

imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present')

print('=====contoh 2')

imgatt2.head()
print(imgatt2.head())
imgatt2.shape
print(imgatt2.shape)


#%%
# melihat apakah burung itu termasuk ke dalam spesies mana

imglabels = pd.read_csv("data/CUB_200_2011/image_class_labels.txt", sep=' ', header=None, names=['imgid', 'label'])

imglabels = imglabels.set_index('imgid')

imglabels.head()
print(imglabels.head())
imglabels.shape
print(imglabels.shape)


#%%

# menjoinkan data

df = imgatt2.join(imglabels)
df = df.sample(frac=1)

# drop label yang depan, dan gunakan label yang paling belakang

df_att = df.iloc[:, :312]
df_label = df.iloc[:, 312:]

df_att.head()
df_label.head()

df_train_att = df_att[:8000]
df_train_label = df_label[:8000]
df_test_att = df_att[8000:]
df_test_label = df_label[8000:]

df_train_label = df_train_label['label']
df_test_label = df_test_label['label']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=100)

clf.fit(df_train_att, df_train_label)

clf.predict(df_train_att.head())

print(clf.score(df_test_att, df_test_label))

#%%

# [2] Confusion Matrix

print("=====Confusion Matrix=====")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, pred_labels)

print(cm)

print('function plot_confusion_matrix')

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes, 
                          normalize=False, 
                          title='Confusion Matrix', 
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #       plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print('Membaca file classes.txt')

birds = pd.read_csv('data/CUB_200_2011/classes.txt', sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']
print(birds)

import numpy as np

np.set_printoptions(precision=2)
plt.figure(figsize=(60,60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.show()

print(plt.show())

#%%

"""
    Mencoba dengan metode Decission Tree dan SVM
"""

# Menggunakaan Dession tree

from sklearn import tree

clftree = tree.DecisionTreeClassifier()
clftree.fit(df_train_att, df_train_label)
clftree.score(df_test_att, df_test_label)
print('===ini menggunakan decision tree')
print(clftree.score(df_test_att, df_test_label))

# Menggunakan SVM
from sklearn import svm

clfsvm = svm.SVC()
clfsvm.fit(df_train_att, df_train_label)
clfsvm.score(df_test_att, df_test_label)
print('===ini menggunakan svm')
print(clfsvm.score(df_test_att, df_test_label))

 
# Pengecekkan Cross Validation
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
# show average score and +/- two standard deviations away ( covering 95% of scores )
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# untuk decission tree
scorestree = cross_val_score(clftree, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

# untuk SVM
scoressvm = cross_val_score(clfsvm, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))


#%%
# pengamatan komponen informasi
max_features_opts = range(5, 50, 5)
n_estimators_opts = range(10, 200, 20)
rf_params = np.empty((len(max_features_opts) * len(n_estimators_opts), 4), float)
i = 0

for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
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
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()
print(plt.show())

#%%









































































