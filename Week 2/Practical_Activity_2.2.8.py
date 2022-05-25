#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 2.2.8

# ### 1. Prepare your workstation

# In[1]:


# Import all the necessary packages.
import numpy as np
import pandas as pd


# ### 2. Import the data set

# In[2]:


# Import the data set.
df = pd.read_csv('breast_cancer_data.csv', 
                 index_col = 'id')

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


# Determine if data set is balanced.
df['diagnosis'].value_counts()


# ### 3. Create a random forest model

# In[7]:


# Import necessary packages.
import imblearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# In[8]:


# Divide data into attributes and labels.
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


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


# In[11]:


# Import 'RandomForestClassifier' package.
from sklearn.ensemble import RandomForestClassifier

# Create a variable to store the classifier:
forest = RandomForestClassifier(n_estimators = 200, criterion = 'gini', 
                                min_samples_split = 2, min_samples_leaf = 2, 
                                max_features = 'auto', bootstrap = True,
                                n_jobs = -1, random_state = 42)

# Train the model with the train subset.
forest.fit(X_train, y_train)

# Predict the response for the test subset.
y_pred = forest.predict(X_test)


# ### 4. Calculate/Check accuracy of model

# In[12]:


# Import necessary packages.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Print the confusion matrix.
print(confusion_matrix(y_test,y_pred))
# Print the classification report.
print(classification_report(y_test,y_pred))
# Print the accuracy score.
print(accuracy_score(y_test, y_pred))


# ### 5. Visualise the random forest model

# In[15]:


# Import necessary packages.
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

# Plot the visualisation:
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5), dpi = 800)
tree.plot_tree(forest.estimators_[0], filled = True)

plt.show()


# In[ ]:




