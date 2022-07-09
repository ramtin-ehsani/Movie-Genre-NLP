#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data = pd.read_csv("drive/MyDrive/nlp_data/raw/cleaned_data.csv")
data


# In[ ]:


data["genre"] = data["genre"].str.split(',')


# In[ ]:


data["genre"][:5]


# In[ ]:


train_df, test_df = train_test_split(
    data,
    test_size=0.2,
)
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in test set: {len(test_df)}")


# In[ ]:


terms = tf.ragged.constant(train_df["genre"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()


def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


print("Vocabulary:\n")
print(vocab)


# In[ ]:


sample_label = train_df["genre"].iloc[0]
print(f"Original label: {sample_label}")

label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")


# In[ ]:


batch_size = 128
auto = tf.data.AUTOTUNE

def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["genre"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["preprocessed_plot"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


# In[ ]:


train_dataset = make_dataset(train_df, is_train=True)
test_dataset = make_dataset(test_df, is_train=False)


# In[ ]:


text_batch, label_batch = next(iter(train_dataset))

for i, text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Plot: {text}")
    print(f"Label(s): {invert_multi_hot(label[0])}")
    print(" ")


# In[ ]:


vocabulary = set()
train_df["preprocessed_plot"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)


# In[ ]:


text_vectorizer = layers.TextVectorization(
    max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf"
)

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

train_dataset = train_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)

test_dataset = test_dataset.map(
    lambda text, label: (text_vectorizer(text), label), num_parallel_calls=auto
).prefetch(auto)


# In[ ]:


model = Sequential()
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(lookup.vocabulary_size(), activation='sigmoid'))


# In[ ]:


model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)


# In[ ]:


history = model.fit(train_dataset, epochs=20)


# In[ ]:


print(model.summary())


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)


# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.xlabel("Epochs")
plt.ylabel('loss')
plt.title("Train {} Over Epochs".format('loss'), fontsize=14)
plt.legend()
plt.grid()
plt.show()


# In[ ]:


plt.plot(history.history['categorical_accuracy'], label='categorical_accuracy')
plt.xlabel("Epochs")
plt.ylabel('categorical_accuracy')
plt.title("Train {} Over Epochs".format('categorical_accuracy'), fontsize=14)
plt.legend()
plt.grid()
plt.show()


# In[ ]:


_, categorical_acc = model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(categorical_acc * 100, 2)}%.")


# In[ ]:




