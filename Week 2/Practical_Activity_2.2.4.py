#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 2.2.4

# ### 1. Prepare your workstation

# In[1]:


# import all the necessary packages
import numpy as np
import pandas as pd


# ### 2. Import the data set

# In[3]:


# Import the data set.
df = pd.read_csv("breast_cancer_data.csv")

# View the DataFrame.
df.info()


# In[4]:


# Determine null values.
df.isnull().sum()


# In[5]:


# Descriptive statistics.
df.describe()


# In[6]:


# Drop null values.
df.drop(labels = 'Unnamed: 32', axis = 1, inplace = True)


# In[7]:


# Determine if the data set is balanced.
df['diagnosis'].value_counts()


# ### 3. Create a decision tree model

# In[8]:


# Import necessary packages.
import imblearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# In[9]:


# Specify the variables.
target_col = 'diagnosis'
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols]
y = df[target_col]


# In[10]:


# Split the data set into train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                    random_state = 1)


# In[12]:


# Import 'DecisionTreeClassifier'.
from sklearn.tree import DecisionTreeClassifier

# Create DecisionTreeClassifier object.
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)

# Train DecisionTreeClassifier.
dtc.fit(X_train, y_train)

# Predict the response for the test data.
y_pred = dtc.predict(X_test)


# ### 4. Calculate accuracy of model

# In[13]:


# Import necessary libraries.
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Print the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)


# In[14]:


# Print the classification report.
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### 5. Plot the decision tree

# In[16]:


# Import matplotlib and 'tree' to create visualisation.
import matplotlib.pyplot as plt
from sklearn import tree

# Plot the decision tree model.
fig, ax = plt.subplots(figsize = (10, 10))
tree.plot_tree(dtc, fontsize = 10)

plt.show()


# In[ ]:




