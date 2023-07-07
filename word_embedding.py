# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 01:38:11 2023

@author: ayush
"""

from tensorflow.keras.preprocessing.text import one_hot
sent=[ 'the glass of milk',
       'the glass of juice',
       'the cup of tea',
       'I am a good boy',
       'I am a good developer',
       'understand the meaning of words',
       'your videos are good',]

# The size of the vocabulary
voc_size=10000

onehot_rep=[one_hot(words, voc_size)for words in sent]


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sent_length=8
embedded_docs=pad_sequences(onehot_rep, padding='pre',maxlen=sent_length)

dim=15


model=Sequential()
model.add(Embedding(voc_size, 10, input_length=sent_length))
model.compile('adam', 'mse')

model.summary()

print(model.predict(embedded_docs))

print(model.predict(embedded_docs)[0])


