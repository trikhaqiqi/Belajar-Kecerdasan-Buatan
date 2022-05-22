# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 20:08:46 2022

@author: trikhaqiqi
"""

# Loading an example dataset

from sklearn import datasets 
# digunakan untuk memanggil class dataset

iris = datasets.load_iris()
# menggunakan contoh dari dataset iris

digits = datasets.load_digits()
# menyimpan nilai data sets digits

#print(digits.images[0])

# menampilkan hasil dari variabel digits





# Learning and predicting

"""
Dalam kasus kumpulan data digit, tugasnya adalah memprediksi, 
dengan diberikan gambar, digit mana yang diwakilinya.
"""

from sklearn import svm 
# untuk mengimport class svm dari sklearn

clf = svm.SVC(gamma=0.001, C=100.)
# Memasukkan implementasi dari "Support Vector Classification" ke variabel clf

clf.fit(digits.data[:-1], digits.target[:-1])
# untuk melakukan pengiriman data training set ke method fit
# method fit bertujuan untuk melatih model

# print(clf.fit(digits.data[:-1], digits.target[:-1]))

clf.predict(digits.data[-1:])
# Untuk melakukan prediksi nilai pada digits

#print(clf.predict(digits.data[-1:]))

from sklearn import datasets

import matplotlib.pyplot as plt

# Load the digits dataset
digits = datasets.load_digits()

# Display the last digit
plt.figure(1, figsize=(3, 3))
# figure adalah object matplotlib yang mengandung semua elemen dari sebuah grafik

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
# interpolasi methode untuk menghasilkan titik di antara titik-titik tertentu
# digunakan untuk menampilkan data sebagai gambar

plt.show()





# Model Persistence

"""
Setelah melatih model scikit-learn, 
diinginkan untuk memiliki cara untuk mempertahankan model 
untuk penggunaan di masa mendatang tanpa harus melatih ulang
"""

from sklearn import svm 
from sklearn import datasets

clf = svm.SVC()
# digunakan untuk memberikan nilai gama secara manual

X, y = datasets.load_iris(return_X_y=True)
# mengambil data sets iris

clf.fit(X, y)
# print(clf.fit(X, y))

# clf sebagai classfilter

import pickle
# untuk mengambil library pickle
# yang memiliki fungsi untuk menyimpan dan membaca data ke dalam /dari sebuah file

s = pickle.dumps(clf)
# untuk membuat variabel s sebagai classifier
# dan dumps ini digunakan ketika harus disimpan dalam file

clf2 = pickle.loads(s)
# variabel clf2 sebagai load

clf2.predict(X[0:1])
# print(clf2.predict(X[0:1]))
# untuk memprediksi

y[0]
# print(y)

from joblib import dump, load
# mengambil dump, melalui library joblib

dump(clf, 'filename.joblib')
# menyimpan model kedalam filename.joblib

clf = load('filename.joblib')
# memanggilnya lagi






# Conventions

"""
scikit-learn estimator mengikuti aturan tertentu untuk membuat 
perilaku mereka lebih prediktif
"""

import numpy as np
from sklearn import random_projection

# mengambil library

rng = np.random.RandomState(0)
# membuat variabel rng, dan mendifinisikan np, fungsi random dan attr random

X = rng.rand(10, 2000) # untuk membuat variabel X dan menentukan nilai X
X = np.array(X, dtype='float32') 

# untuk menyimpan nilai random dengan type data float 32

X.dtype # untuk mengubah type data menjadi float64
# print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
# untuk membuat variabel transformer dan mendefinisikan nilai random projection

X_new = transformer.fit_transform(X)
# membuat variabel x yang baru

X_new.dtype # memanggil datatype
# print(X_new.dtype) 
# menampilkan data type






























# "Support Vector Classification" 
# digunakan untuk mengklasifikasi

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)
# print(clf.fit(iris.data, iris.target))

list(clf.predict(iris.data[:3]))
# print(list(clf.predict(iris.data[:3])))

clf.fit(iris.data, iris.target_names[iris.target])
# print(clf.fit(iris.data, iris.target_names[iris.target]))

list(clf.predict(iris.data[:3]))
# print(list(clf.predict(iris.data[:3])))









