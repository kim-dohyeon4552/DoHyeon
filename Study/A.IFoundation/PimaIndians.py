#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np


# In[207]:


dataset = np.loadtxt("pima-indians-diabetes.csv",delimiter = ",")


# In[291]:


x_train = np.array(dataset[:650,0:8])
y_train = np.array(dataset[:650,8:])
x_val = np.array(dataset[650:,0:8])
y_val = np.array(dataset[650:,8:])
x_test = np.array(dataset[700:,0:8])
y_test = np.array(dataset[700:,8:])


# In[292]:


y_train = utils.to_categorical(y_train)
y_val = utils.to_categorical(y_val)
y_test = utils.to_categorical(y_test)


# In[293]:


model = Sequential()


# In[294]:


model.add(Dense(units=20, input_dim=8, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2,  activation='sigmoid'))


# In[295]:


# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[296]:


# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=0, validation_data=(x_val, y_val))


# In[297]:


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=30)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))

