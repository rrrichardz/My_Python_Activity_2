#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 2.3.5

# ### 1. Prepare your workstation

# In[10]:


# Import all the necessary packages.
import numpy as np
import pandas as pd

import warnings  
warnings.filterwarnings("ignore")


# In[11]:


# Import data file.
df_ais = pd.read_csv('ais.csv')

df_ais.info()


# In[12]:


# Determine null values.
df_ais.isnull().sum()


# ### 2. Evaluate the variables

# In[13]:


# Descriptive analysis.
df_ais.describe()


# ### 3. Drop unneeded columns

# In[14]:


# Drop unneedd columns.
# Quick analysis on the variable.
print(len(df_ais['sex'].unique()))
print(len(df_ais['sport'].unique()))

# In cluster analysis we cannot use unique identifier so we drop this column.
df_ais.drop('sex', axis = 1, inplace = True)


# In[15]:


# Display the column names.
df_ais.columns


# In[16]:


# Plot a countplot to display the frequency per sport type.
import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize = (12, 12))
ax = sns.countplot(x = "sport", data = df_ais)
plt.title('Blood characteristics of athletes')
plt.xlabel('Sport type')
plt.ylabel('Frequency')

for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()),
                (p.get_x() + 0.1, p.get_height() + 50),
                va = 'center')


# ### 4. Specify the target variable 

# In[17]:


# Define the target variables.
X = df_ais
y = df_ais['sport']

# Import the LabelEncoder class:
from sklearn.preprocessing import LabelEncoder

# Convert the target variable to integers.
le = LabelEncoder()

# Replace in the existing DataFrame with the integer values.
X['sport'] = le.fit_transform(X['sport'])
y = le.transform(y)


# In[19]:


# View the modified DataFrame.
X.info()


# In[20]:


X.head()


# ### 5. Normalise the data set

# In[21]:


# Create an list with the column labels from X:
x_cols = X.columns

# Import the MinMaxScaler class.
from sklearn.preprocessing import MinMaxScaler 

# Create the object from ‘MinMaxScaler’.
ms = MinMaxScaler() 
# Modify X to scale values between 0 and 1.
X = ms.fit_transform(X) 
# Set X as equal to a new DataFrame.
X = pd.DataFrame(X, columns=[x_cols]) 

# Check the contents of the modified DataFrame.
X.head() 


# ### 6. Apply the clustering algorithm

# In[22]:


# Import KMeans class.
from sklearn.cluster import KMeans

# Apply the clustering and fit() method.
kmeans = KMeans(n_clusters=2, random_state=0) 

kmeans.fit(X)


# In[23]:


# Indicate 'kmeans()' applies to 'cluster_centers'.
kmeans.cluster_centers_


# In[24]:


# Check the inertia for the data set.
kmeans.inertia_


# ### 7. Evaluate the output

# In[25]:


# Extract the labels from the k-means.
labels = kmeans.labels_

# Check correctly labelled instances.
correct_labels = sum(y == labels)

# Print the output.
print('Result: %d out of %d samples were correctly labelled.' % (correct_labels,
                                                                y.size))


# ### 8. Improve the accuracy (elbow method)

# In[29]:


# Create an empty list.
cs = []

# Employ a loop to test cluster sizes:
for i in range(1, 11):
    # Create object k-means.
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                   max_iter = 300, n_init = 100,
                   random_state = 0)
    # Apply the fit() method.
    kmeans.fit(X)
    # Add the inertia value.
    cs.append(kmeans.inertia_)

# Create a plot.
plt.plot(range(1, 11), cs)
# Speciy the title, x-axis label and y-axis label.
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')

# Display the plot.
plt.show()

# From the elbow method we can check the optimal number from 3 - 6.


# In[40]:


# Create a k-means object with six clusters:
kmeans = KMeans(n_clusters = 6, random_state = 0)

# Apply 'fit()', using the DataFrame, to the k-means object.
kmeans.fit(X)

# Check how many of the samples were corectly labelled:
labels = kmeans.labels_
correct_labels = sum(y == labels)

# Display the accuracy score:
print('Result: %d out of %d samples were correctly labelled.' % (correct_labels,
                                                                y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels / float(y.size)))


# ### 9. Visualise the clusters

# In[43]:


# Create the figure area.
fig = plt.figure(figsize = (26, 6))

# Create a 3D projection area.
ax = fig.add_subplot(131, projection = '3d')

# Create a 3D scatter plot and specify the data source for each axis:
ax.scatter(df_ais['ht'], df_ais['wt'], df_ais['lbm'],
          c = labels, s = 15)

# [4] Set the label for each dimension:
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Lean Body Mass')

# [5] Show the plot.
plt.show()


# In[44]:


# Create the figure area.
fig = plt.figure(figsize = (26, 6))

# Create a 3D projection area.
ax = fig.add_subplot(131, projection = '3d')

# Create a 3D scatter plot and specify the data source for each axis:
ax.scatter(df_ais['rcc'], df_ais['wcc'], df_ais['hg'],
          c = labels, s = 15)

# [4] Set the label for each dimension:
ax.set_xlabel('Red Blood Cell Count')
ax.set_ylabel('White Blood Cell Count')
ax.set_zlabel('Hemoglobin Concentration')

# [5] Show the plot.
plt.show()


# In[45]:


# Create the figure area.
fig = plt.figure(figsize = (26, 6))

# Create a 3D projection area.
ax = fig.add_subplot(131, projection = '3d')

# Create a 3D scatter plot and specify the data source for each axis:
ax.scatter(df_ais['bmi'], df_ais['ht'], df_ais['wt'],
          c = labels, s = 15)

# [4] Set the label for each dimension:
ax.set_xlabel('Body Mass Index')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')

# [5] Show the plot.
plt.show()


# In[ ]:




