# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:53:45 2023

@author: ayush
"""

import pandas as pd

df=pd.read_csv('train.csv')

# Dropping the Nan values
df=df.dropna()

X=df.drop('label', axis=1)
y=df['label']

import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

voc_size=5000

messages=X.copy()
messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords
#nltk.download('stopwords')

### Dataset Preprocessing
from nltk.stem import WordNetLemmatizer
lm= WordNetLemmatizer()
corpus=[]
for i in range(0, len(messages)):
    print(i)
    review=re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review= review.lower()
    review= review.split()
    
    review= [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
    
    
onehot_rep=[one_hot(words, voc_size) for words in corpus]


sent_length=20
embedded_docs= pad_sequences(onehot_rep, padding='pre', maxlen=sent_length)


###Creating the model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())



import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_final, y_final, test_size=0.2, random_state=42)


##Training the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


### Adding a Drop-out layer
from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


y_pred= (model.predict(X_test)>0.5).astype("int32")
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)


ds=pd.read_csv('test.csv')
ds=ds.dropna()

D=ds

submission=D.copy()
submission.reset_index(inplace=True)
lorpus=[]
for i in range(0, len(submission)):
    print(i)
    rev=re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    rev= rev.lower()
    rev= rev.split()
    
    rev= [lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    rev= ' '.join(rev)
    lorpus.append(rev)
    
    
onehot_reps=[one_hot(words, voc_size) for words in lorpus]

embedded_doc= pad_sequences(onehot_reps, padding='pre', maxlen=sent_length)

D_final=np.array(embedded_doc)
y_sub= (model.predict(D_final)>0.5).astype("int32")


