#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 2.1.12

# ### 1. Prepare your workstation

# In[1]:


# import all the necessary packages
import numpy as np
import pandas as pd


# ### 2. Import the data set

# In[2]:


# Import data set into Python.
df = pd.read_csv("breast_cancer_data.csv")

df.info()


# In[3]:


# Determine null values.
df.isnull().sum()


# In[4]:


# Descriptive statistics.
df.describe()


# In[5]:


# Drop null values.
df.drop(labels = 'Unnamed: 32', axis = 1, inplace = True)


# In[6]:


# Determine if the data set is balanced.
df['diagnosis'].value_counts()


# ### 3. Create an SVM model

# In[7]:


# Import necessary packages.
import imblearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# In[8]:


# Specify the variables.
target_col = 'diagnosis'
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols]
y = df[target_col]


# In[9]:


# Split the data into train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 1)


# In[ ]:


# Import 'svm' package from sklearn.
from sklearn import svm

# Create a svm classifier (linear kernel).
clf = svm.SVC(kernel = 'linear', gamma = 'scale')

# Train the model using the train subset.
clf.fit(X_train, y_train)

# Predict the response for the test subset.
y_pred = clf.predict(X_test)


# ### 4. Calculate accuracy of model

# In[ ]:


# Import necessary packages.
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Print the confusion matrix.
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)


# In[ ]:


# Print classification report.
from sklearn.metrics import classification_report

print(classification_report(y-test, y-pred))


# In[ ]:


# Print the accuracy score.
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:




