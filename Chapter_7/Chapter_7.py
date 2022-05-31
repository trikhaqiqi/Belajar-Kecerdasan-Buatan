# In[1]: import library yang diperlukan
import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

# In[2]: tumpuk 5 data csv kedalam variabel d terus kocok2

d = pd.concat([pd.read_csv("data/Youtube01-Psy.csv"),
               pd.read_csv("data/Youtube02-KatyPerry.csv"),
               pd.read_csv("data/Youtube03-LMFAO.csv"),
               pd.read_csv("data/Youtube04-Eminem.csv"),
               pd.read_csv("data/Youtube05-Shakira.csv")])
d = d.sample(frac=1)

# In[3]: mengecek k fold cross validation
    
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])
print(kfold)
print(splits)

# In[3]: mengecek k fold cross validation
for train, test in splits:
    print("--------------Splits---------------------")
    print(test)



# In[3]: membuat fungsi train dan test

def train_and_test(train_idx, test_idx):
    
    train_content = d['CONTENT'].iloc[train_idx]
    test_content = d['CONTENT'].iloc[test_idx]
    
    tokenizer = Tokenizer(num_words=2000)
    
    # learn the training words (not the testing words!)
    tokenizer.fit_on_texts(train_content)

    # options for mode: binary, freq, tfidf
    d_train_inputs = tokenizer.texts_to_matrix(train_content, mode='tfidf')
    d_test_inputs = tokenizer.texts_to_matrix(test_content, mode='tfidf')

    # divide tfidf by max
    d_train_inputs = d_train_inputs/np.amax(np.absolute(d_train_inputs))
    d_test_inputs = d_test_inputs/np.amax(np.absolute(d_test_inputs))

    # subtract mean, to get values between -1 and 1
    d_train_inputs = d_train_inputs - np.mean(d_train_inputs)
    d_test_inputs = d_test_inputs - np.mean(d_test_inputs)

    # one-hot encoding of outputs
    d_train_outputs = np_utils.to_categorical(d['CLASS'].iloc[train_idx])
    d_test_outputs = np_utils.to_categorical(d['CLASS'].iloc[test_idx])

    model = Sequential()
    model.add(Dense(512, input_shape=(2000,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])

    model.fit(d_train_inputs, d_train_outputs, epochs=10, batch_size=16)

    scores = model.evaluate(d_test_inputs, d_test_outputs)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores


# In[4]: mengecek k fold cross validation
    
kfold = StratifiedKFold(n_splits=5)
splits = kfold.split(d, d['CLASS'])
cvscores = []
for train_idx, test_idx, in splits:
    scores = train_and_test(train_idx, test_idx)
    cvscores.append(scores[1] * 100)
# In[4]: print rata2 dari skore akurasi
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

