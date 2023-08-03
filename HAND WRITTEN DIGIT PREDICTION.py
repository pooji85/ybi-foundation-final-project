#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# # IMPORTING DATA

# In[4]:


from sklearn.datasets import load_digits


# In[5]:


df=load_digits()


# In[6]:


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, df.images, df.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


# # DATA PREPROCESSING

# In[7]:


df.images.shape


# In[8]:


df.images[0]


# In[9]:


df.images[0].shape


# In[10]:


len(df.images)


# In[11]:


n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))


# In[12]:


data[0]


# In[13]:


data[0].shape


# In[14]:


data.shape


# # SCALING IMAGE DATA

# In[15]:


data.min()


# In[16]:


data.max()


# In[17]:


data = data/16


# In[18]:


data.min()


# In[19]:


data.max()


# In[20]:


data[0]


# # TRAIN TEST SPLIT DATA

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(data, df.target, test_size=0.3)


# In[23]:


x_train.shape, x_test.shape,y_train.shape,x_test.shape


# # RANDOM FOREST MODEL

# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


rf=RandomForestClassifier()


# In[26]:


rf.fit(x_train,y_train)


# # PREDICT TEST DATA

# In[27]:


y_pred = rf.predict(x_test)


# In[28]:


y_pred


# # MODEL ACCURACY 

# In[29]:


from sklearn.metrics import confusion_matrix, classification_report


# In[30]:


confusion_matrix(y_test, y_pred)


# In[31]:


print(classification_report(y_test, y_pred))


# In[ ]:




