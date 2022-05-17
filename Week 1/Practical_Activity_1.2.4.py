#!/usr/bin/env python
# coding: utf-8

# ## Practical_Activity_1.2.4

# ### 1. Prepare your workstation

# In[14]:


# Import the necessary libraries.
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 2. Import the data set

# In[3]:


# Import the data set.
df_test = pd.read_csv("loyalty_club.csv")

df_test


# ### 3. Define the variables

# In[9]:


# Identify dependent variable.
y = df_test['Yearly Amount Spent']

# Identify independent variable.
x = df_test['Length of Membership']

# Check for linearity.
plt.scatter(x, y)


# ### 4. Run an OLS test

# In[15]:


# Create formula and pass through OLS method:
f = 'y ~ x'
test = ols(f, data = df_test).fit()

# Print the regression table:
test.summary()


# ### 5. Create linear equation

# In[18]:


# # Set the x coefficient to '64.2187' and the constant to '272.3998'
# to generate the regression table:

y_pred = 64.2187 * x + 272.3998

y_pred


# ### 6. Plot the regression

# In[19]:


# Plot the data points:
plt.scatter(x, y)

# Plot the regression line
plt.plot(x, y_pred, color = 'red')


# In[ ]:




