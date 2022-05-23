#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 2.1.7

# ### 1. Prepare your workstation

# In[2]:


# import all the necessary packages
import numpy as np
import pandas as pd


# ### 2. Import the data set

# In[3]:


# Import data file into Python.
df = pd.read_csv("breast_cancer_data.csv", index_col = 'id')

df.info()


# In[4]:


# Determine null values.
df.isnull().sum()


# In[5]:


# Descriptive statistics.
df.describe()


# In[6]:


# Drop all null values.
df.drop(labels = 'Unnamed: 32', axis = 1, inplace = True)


# In[8]:


# Determine if data set is balanced.
df['diagnosis'].value_counts()


# ### 3. Create a BLR model

# In[9]:


# Import necessary packages.
import imblearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# In[13]:


# Specify the variables.
target_col = 'diagnosis'
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols]
y = df[target_col]


# In[15]:


# Split the data set into train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 1)


# In[17]:


# Create the model.
logreg_model = LogisticRegression()

logreg_model.fit(X_train, y_train)


# In[21]:


# Calculate the predicted labels and predicted probabilities on the test set.
# Predict test class:
y_pred = logreg_model.predict(X_test)

# Predict test probabilities:
y_pp = logreg_model.predict_proba(X_test)


# ### 4. Calculate accuracy of model

# In[19]:


# Create the confusion matrix for your classifier's performance on the test set.
con_mat = confusion_matrix(y_test, y_pred, labels=['M', 'B'])


# In[23]:


# Predicting cancer based on some kind of detection measure.
confusion = pd.DataFrame(con_mat, index = ['predicted_cancer',
                                           'predicted_healthy'],
                        columns = ['is_cancer', 'is_healthy'])

confusion


# In[25]:


# Print the metrics.
print(metrics.accuracy_score(y_test, y_pred))


# In[26]:


# Print the confusion matrix.
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)


# In[27]:


# Print classification report.
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:




