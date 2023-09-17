#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import load_diabetes  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error, r2_score 


# In[9]:


# Loading the sklearn diabetes dataset  
X, Y = load_diabetes(return_X_y=True)  


# In[10]:


# Taking only one feature to perform simple linear regression  
X = X[:,8].reshape(-1,1) 


# In[11]:


# Splitting the dependent and independent features of the dataset into training and testing dataset  
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 10 )  


# In[14]:


# Creating an instance for the linear regression model of sklearn  
lr = LinearRegression() 


# In[15]:


lr.fit(X_train,Y_train)


# In[16]:


Y_pred=lr.predict(X_test)
X_pred=lr.predict(X_train)


# In[20]:


plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,X_pred,color='red')
plt.title("Simple Linear Regression")
plt.xlabel("Target Values")
plt.ylabel("Independent Features")


# In[22]:


mean_squared_error(Y_test,Y_pred)


# In[31]:


r2_score(Y_test,Y_pred)*100


# In[ ]:




