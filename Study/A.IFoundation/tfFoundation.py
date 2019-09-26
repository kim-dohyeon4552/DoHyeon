#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
print(tf.__version__)
import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np


# In[19]:


# 1. 데이터셋 준비하기
X_train = np.array(
[
    1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
]
)

Y_train = np.array(
[
    3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27
])

X_val = np.array(
[
    1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
])

Y_val = np.array(
[
    3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,3,6,9,12,15,18,21,24,27,
])


# In[20]:


print(X_train.shape)
print(X_train)
print(Y_train.shape)
print(Y_train)


# In[21]:


# 라벨링 전환
Y_train = utils.to_categorical(Y_train,28)
Y_val = utils.to_categorical(Y_val,28)


# In[22]:


print(X_train)
print(Y_train)
print(Y_train.shape)
print(Y_val)
print(Y_val.shape)


# In[31]:


model = Sequential()
model.add(Dense(units=56, input_dim=1, activation='elu'))
model.add(Dense(units=28,  activation='softmax'))


# In[32]:


# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[33]:


# 4. 모델 학습시키기
hist = model.fit(X_train, Y_train, epochs=200, batch_size=1, verbose=1, validation_data=(X_val, Y_val))


# In[34]:


# 5. 모델 학습 과정 표시하기
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


# In[44]:


# 6. 모델 사용하기
X_test = np.array([
    1,2,3,4,5,6,7,8,9
])
Y_test = np.array([
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
])
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=1)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))


# In[51]:


width = 28
height = 28
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width*height).astype('float32') / 255.0
x_test= x_test.reshape(10000, width*height).astype('float32') / 255.0


# In[54]:


x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]


# In[56]:


y_train = utils.to_categorical(y_train)
y_val = utils.to_categorical(y_val)
y_test = utils.to_categorical(y_test)

