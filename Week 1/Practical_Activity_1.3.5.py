#!/usr/bin/env python
# coding: utf-8

# ## Practical_Activity_1.3.5

# ### 1. Prepare your workstation

# In[1]:


# Import all the necessary packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
import sklearn

from sklearn import datasets 
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols


# ### 2. Import the data set

# In[5]:


# Import the data set.
df_ecom = pd.read_csv("Ecommerce_data.csv")

# View DataFrame.
df_ecom.info()


# ### 3 Define variables

# In[6]:


# Define dependent variable.
y = df_ecom['Median_s']

# Define independent variable.
X = df_ecom[['avg_no_it', 'tax']]


# In[7]:


# Create train and test data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 5)


# In[8]:


# Run regression on training data.
multi = LinearRegression()
multi.fit(X_train, y_train)


# In[9]:


multi.predict(X_train)


# In[12]:


# Check the value of the R-squared, intercept and coefficients.
print('R-squared: ', multi.score(X_train, y_train))
print('Intercept: ', multi.intercept_)
print('Coefficients:')
list(zip(X_train, multi.coef_))


# In[13]:


# Make predictions.
New_Value1 = 5.75
New_Value2 = 15.2
print ('Predicted Value: \n', multi.predict([[New_Value1 ,New_Value2]]))


# ### 4. Check the model with OLS

# In[18]:


# Run regression on test subset.
mlr = LinearRegression()  
mlr.fit(X_test, y_test)


# In[19]:


model = sm.OLS(y_test, sm.add_constant(X_test)).fit()
y_pred = model.predict(sm.add_constant(X_test))
print_model = model.summary()

print(print_model)


# In[20]:


# Predictions on the test subset.
y_pred_mlr = mlr.predict(X_test)
print('Prediction for test set: {}'.format(y_pred_mlr))


# In[21]:


print(mlr.score(X_test, y_test) * 100)


# In[22]:


# Determine the R-squared, mean absolute error and mean square error.
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)

print('R-squared: {:.2f}'.format(mlr.score(X, y) * 100))
print('Mean Absolute Error: ', meanAbErr)
print('Mean Squared Error: ', meanSqErr)


# In[24]:


# Check for multicollinearity.
x_temp = sm.add_constant(X_train)

# Create an empty DataFrame.
vif = pd.DataFrame()

# Calculate the VIF.
vif['VIF Factor'] = [variance_inflation_factor(x_temp.values, i) for i in range (x_temp.values.shape[1])]

# Create the feature column.
vif['feature'] = x_temp.columns

# Print the values to 2 decimal places.
print(vif.round(2))


# In[27]:


# Check for homoscedasticity.
model = sms.het_breuschpagan(model.resid, model.model.exog)


# In[28]:


# Print the results of the Breusch-Pagan test:
terms = ['LM stat', 'LM Test p-value', 'F-stat', 'F-test p-value']
print(dict(zip(terms, model)))


# In[ ]:




