# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:32:20 2023

@author: ayush
"""
import pandas as pd
df = pd.read_csv('train.csv')

X= df.drop('label', axis=1)
y= df['label']

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
df= df.dropna()

messages= df.copy()
messages.reset_index(inplace=True)

messages['title'][6]


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps= PorterStemmer()
corpus= []
for i in range(0,len(messages)):
    review= re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review= review.lower()
    review= review.split()
    
    review= [ps.stem(word)for word in review if not word in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
    

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000,ngram_range=(1,3))
X= cv.fit_transform(corpus).toarray()

y= messages['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

cv.get_feature_names_out()[:20]
cv.get_params()

from sklearn.naive_bayes import MultinomialNB
classifier= MultinomialNB()


import matplotlib.pyplot as plt


from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train, y_train)
# Making a Pickle file fo our Model
import pickle
pickle.dump(classifier,open("model.pkl","wb"))
y_pred= classifier.predict(X_test)
score= metrics.accuracy_score(y_test, y_pred)
print("accuracy: %0.3f" % score)
cm= metrics.confusion_matrix(y_test, y_pred)
#plot_confusion_matrics(cm, classes=['FAKE', 'REAL'])


    
