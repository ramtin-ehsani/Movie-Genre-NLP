#!/usr/bin/env python
# coding: utf-8

# # CNN Method

# In[20]:


from numpy import array
import keras
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Bidirectional
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from numpy import array
from numpy import asarray
from numpy import zeros
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile


# In[21]:


col_list = ["preprocessed_plot","action", "sci-fi", "comedy", "horror","drama","animation",
            "mystery","crime","fantasy","thriller","romance","adventure","biography"]
meta = pd.read_csv("data_for_bert.csv", usecols=col_list)
meta.head()


# In[22]:


train_set_labels = meta[["action", "sci-fi", "comedy", "horror","drama","animation",
            "mystery","crime","fantasy","thriller","romance","adventure","biography"]]
train_set_labels.head()


# In[23]:


X = list(meta["preprocessed_plot"])
y = train_set_labels.values


# In[24]:


print(len(X))


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ## Download only ONCE

# In[10]:


url = 'http://nlp.stanford.edu/data/glove.6B.zip'
r = requests.get(url, allow_redirects=True)

open('glove.6B.zip', 'wb').write(r.content)

path_to_zip_file = 'glove.6B.zip'
directory_to_extract_to = 'glove.6B.txt'


# In[11]:


with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)


# In[26]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# In[27]:


embeddings_dictionary = dict()

glove_file = open('glove.6B.txt/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[28]:


input = tf.keras.layers.Input(shape=(maxlen,))
x = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input)

x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)

avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

x = tf.keras.layers.concatenate([avg_pool, max_pool])

preds = tf.keras.layers.Dense(13, activation="sigmoid")(x)

model = tf.keras.Model(input, preds)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['acc'])


# In[29]:


model.summary()


# In[34]:


batch_size = 128

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size,
          epochs=10, verbose=1)


# In[35]:


score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[36]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[ ]:




