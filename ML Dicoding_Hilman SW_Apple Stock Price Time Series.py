#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[1]:


import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ## Load Datasets

# In[2]:


df = pd.read_csv('D:\\Kumpulan Dataset\\Kaggle\\AAPL.csv')
df.head()


# ## Dataset Information

# In[3]:


df.info()


# In[4]:


df['Date'] = pd.to_datetime(df['Date'])
df.info()


# In[5]:


df.count()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df = df.drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis='columns')


# In[10]:


df.head()


# In[11]:


df.info()


# ## Time Series Visualization

# In[12]:


dates = df['Date'].values
temp  = df['Open'].values
 
plt.figure(figsize=(20,5))
plt.plot(dates, temp)
plt.title('AAPL Open Value', fontsize=20)


# ## Modeling

# In[13]:


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(temp, dates, test_size = 0.2, random_state = 0 , shuffle=False)


# In[15]:


data_x_train = windowed_dataset(X_train, window_size=60, batch_size=100, shuffle_buffer=5000)
data_x_test = windowed_dataset(X_test, window_size=60, batch_size=100, shuffle_buffer=5000)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])


# In[16]:


max_val = df['Open'].max()
print('Max value = ', max_val)


# In[17]:


min_val = df['Open'].min()
print('Min value = ', min_val)


# In[18]:


mae = (max_val-min_val) * (10/100)
print("MAE = ", mae)


# In[19]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < 10):
      self.model.stop_training = True
      print("\nMAE of the model < 10% of data scale")
callbacks = myCallback()


# In[20]:


tf.keras.backend.set_floatx('float64')
history = model.fit(data_x_train, epochs=500, validation_data=data_x_test, callbacks=[callbacks])


# In[21]:


fig = plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

