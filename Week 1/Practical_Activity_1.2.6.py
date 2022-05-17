#!/usr/bin/env python
# coding: utf-8

# ## Practical_Activity_1.2.6

# ### 1. Prepare your workstation

# In[20]:


# Import all the necessary packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn

from sklearn import datasets 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# ### 2. Import the data set

# In[22]:


# Import the data set.
df = pd.read_csv("Ecommerce_data.csv")

# View DataFrame.
df.head()


# In[24]:


# Fill the missing values to 0.
df.fillna(0, inplace = True)


# ### 3. Define the variables

# In[25]:


# Define the variables.
x = df['avg_no_it'].values.reshape(-1, 1)
y = df['Median_s'].values


# ### 4. Split the data set

# In[30]:


# Split the data into training = 70% and test = 30% subsets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,
                                                   random_state = 0)


# ### 5. Run a linear equation

# In[31]:


# Linear regression.
lr = LinearRegression()


# In[32]:


# Fit the data into the model.
lr.fit(x_train, y_train)


# In[33]:


# Predict the x-test.
y_pred = lr.predict(x_test)


# ### 6. Plot the regression

# In[34]:


# Training data visualisation.
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'green')
plt.title("Avg. no of items vs Median of Seller Business(Training Data)")
plt.xlabel("Avg. no of items")
plt.ylabel("Median of Seller Business")

plt.show()


# In[38]:


# Test data visualisation.
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, y_pred, color = 'green')
plt.title("Avg. no of items vs Median of Seller Business(Training Data)")
plt.xlabel("Avg. no of items")
plt.ylabel("Median of Seller Business")

plt.show()


# ### 7. Print the values

# In[39]:


# Print the R-squared value.
print(lr.score(x_train, y_train))


# In[40]:


# Print the intercept value.
print("Intercept value: ", lr.intercept_)
# Print the coefficient value.
print("Coefficient value: ", lr.coef_)


# In[44]:


# The R-squared tells us that the model is explaining a fraction over 50% of the model.
# The intercept value of -35.99 tells us that as the low stat variable increases by 1,
# the predicted value of Median_s decreases by -35.99.


# In[ ]:




