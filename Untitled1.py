#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 0.2, 4.0, -1.2])
print(softmax(a))
print(np.sum(softmax(a)))


# In[2]:


def softmax(a):
    C = np.max(a)
    return (np.exp(a-C) / np.sum(np.exp(a - C)))

A = np.array([1000, 900, 1050, 500])
print(softmax(A))


# In[3]:


def LeakyReLU(x):
    a = 0.01
    return np.maximum(a*x, x)

x = np.array([0.5, -1.4, 3, 0, 5])
print(LeakyReLU(x))


# In[4]:


def ELU(x):
    alpha = 1.0
    return (x >= 0) * x + (x < 0) * alpha * (np.exp(x) - 1)

print(ELU(4))
print(ELU(np.array([-2, 0.1, 4])))


# In[5]:


def sigmoid(X):
    return 1/ (1 + np.exp(-X))

#Layer1 definition
X = np.array([1.0, 0.5, 0.4])
W1 = np.array([[0.1, 0.3, 0.5], 
               [0.2, 0.4, 0.6], 
               [0.3, 0.5, 0.7]])
B1 = np.array([1, 1, 1])

print(X.shape)
print(W1.shape)
print(B1.shape)


# In[6]:


A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(A1)
print(Z1)


# In[7]:


#Layer2 definition
W2 = np.array([[0.2, 0.4, 0.6], 
               [0.1, 0.3, 0.5], 
               [0.4, 0.6, 0.8]])
B2 = np.array([1, 1, 1])

A2 = np.dot(A1, W2) + B2
Z2 = sigmoid(A2)

print(A2)
print(Z2)


# In[10]:


#Layer3 definition
W3 = np.array([[0.1, 0.3],
               [-0.1, -0.5],
               [0.3, 0.5]])
B3 = np.array([1, 1])

A3 = np.dot(A2, W3) + B3
Z3 = sigmoid(A3)

print(A3)
print(Z3)

#Layer4 definition
W4 = np.array([[0.1, 0.2],
               [0.3, 0.5]])
B4 = np.array([1, 1])

A4 = np.dot(A3, W4) + B4
Y = sigmoid(A4)

print(A4)
print(Y)


# In[ ]:




