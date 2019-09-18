#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
mnist = fetch_openml('mnist_784', data_home='C:\\Users\\dolo8001\\Desktop\\dataset')
N, d = mnist.data.shape


# In[52]:


X_all = mnist.data
y_all = mnist.target


# In[43]:


X0 = X_all[np.where(y_all == '0')[0]] # all digit 0
X1 = X_all[np.where(y_all == '1')[0]] # all digit 1
y0 = np.zeros(X0.shape[0]) # class 0 label
y1 = np.ones(X1.shape[0]) # class 1 label


# In[44]:


X = np.concatenate((X0, X1), axis = 0) # all digits
y = np.concatenate((y0, y1)) # all labels


# In[45]:


# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000)


# In[46]:


model = LogisticRegression(C = 1e5) # C is inverse of lam
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy %.2f %%" % (100*accuracy_score(y_test, y_pred.tolist())))


# In[ ]:





# In[ ]:





# In[ ]:




