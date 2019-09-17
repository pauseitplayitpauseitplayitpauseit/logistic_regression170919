#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np


# In[62]:


def sigmoid(S):
    return 1/(1 + np.exp(-S))
    
def prob(w, X):
    return sigmoid(X.dot(w))


# In[63]:


def loss(w, X, y,lam):
    z = prob(w, X)
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + (1/X.shape[0])*(lam/2)*(w@w.T)


# In[64]:


def logistic_regression(w_init, X, y, lam = 0.001, lr = 0.1, nepoches = 2000):
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)]
    ep = 0
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N)
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w - lr*((zi - yi)*xi + lam*w)
            loss_hist.append(loss(w, X, y, lam))
            if np.linalg.norm(w - w_old)/d < 1e-6:
                break
            w_old = w
        return w, loss_hist


# In[65]:


X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])


# In[87]:


# bias trick
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001
w, loss_hist = logistic_regression(w_init, Xbar, y, lam, lr = 0.05, nepoches = 500)
print(w)
print(loss(w, Xbar, y, lam))


# In[ ]:





# In[ ]:





# In[ ]:




