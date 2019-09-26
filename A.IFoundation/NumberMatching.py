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


# In[4]:


width = 28
height = 28
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width*height).astype('float32') / 255.0
x_test= x_test.reshape(10000, width*height).astype('float32') / 255.0


# In[5]:


x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]


# In[6]:


y_train = utils.to_categorical(y_train)
y_val = utils.to_categorical(y_val)
y_test = utils.to_categorical(y_test)


# In[38]:


model = Sequential()
model.add(Dense(units=1024, input_dim=width*height, activation='elu'))
model.add(Dense(units=512,  activation='elu'))
model.add(Dense(units=512,  activation='elu'))
model.add(Dense(units=128,  activation='elu'))
model.add(Dense(units=64,  activation='elu'))
model.add(Dense(units=10,  activation='softmax'))


# In[43]:


# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[65]:


# 4. 모델 학습시키기
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience=10)
hist = model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1, validation_data=(x_val, y_val),callbacks =[early_stopping])


# In[66]:


loss_and_metrics = model.evaluate(x_test, y_test, batch_size=30)

print('')
print('loss : ' + str(loss_and_metrics[0]))
print('accuray : ' + str(loss_and_metrics[1]))


# In[67]:


yhat_test = model.predict(x_test, batch_size = 32)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt_row = 10
plt_col = 10

plt.rcParams["figure.figsize"] = (15,15)

f, axarr = plt.subplots(plt_row, plt_col)
cnt = 0
i = 0
while cnt < (plt_row*plt_col):
    if np.argmax(y_test[i]) == np.argmax(yhat_test[i]):
        i += 1
        continue
    sub_plt = axarr[int(cnt/plt_row), int(cnt%plt_col)]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i].reshape(width,height))
    sub_plt_title = 'R:' + str(np.argmax(y_test[i]))+'p:'+str(np.argmax(yhat_test[i]))
    sub_plt.set_title(sub_plt_title)
    i += 1
    cnt += 1
plt.show()
print(y_test[1])

