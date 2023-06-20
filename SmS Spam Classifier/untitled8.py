# -*- coding: utf-8 -*-
"""
Created on Tue May 30 01:41:55 2023

@author: ayush
"""

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep='\t',names=["label","message"])
messages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
# lm = WordNetLemmatizer()
corpus= []




for i in range (0, len(messages)):
    rv=re.sub('[^a-zA-Z]', ' ' , messages['message'][i])
    rv=rv.lower()
    rv=rv.split()
    rv=[ps.stem(word) for word in rv if not word in set(stopwords.words('english'))]
    rv= ' '.join(rv)
    corpus.append(rv)
print(corpus)




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X= cv.fit_transform(corpus).toarray()

y= pd.get_dummies(messages['label'])
y=y.iloc[:,1].values






from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2, random_state=0)
 
from sklearn.naive_bayes import MultinomialNB
spam_detector = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detector.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)
print(accuracy)
