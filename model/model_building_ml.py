#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[7]:


df = pd.read_csv("Bank Customer Churn Prediction_dataset.csv")


# In[13]:


df.drop('customer_id', axis=1, inplace=True)
df.drop('gender', axis=1, inplace=True)


# In[8]:


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[9]:


X = pd.get_dummies(X, columns=['country'], prefix=["country"] ) 
X = pd.get_dummies(X, columns=['products_number'], prefix=["products_no"] )


# In[10]:


steps = [('rescale', StandardScaler()),
         ('ranf',RandomForestClassifier(
                               n_estimators=1000, 
                               max_depth=10,
                               max_features=4,
                               min_samples_split=7
         ))]
modelRF = Pipeline(steps)
modelRF.fit(X, y)


# In[11]:


with open('model.pkl', 'wb') as f:
    pickle.dump(modelRF, f)

