#!/usr/bin/env python
# coding: utf-8

# ## Practical_Activity_1.1.6

# ### 1. Prepare your workstation

# In[5]:


# Import the necessary libraries.
import statsmodels.stats.api as sms
from statsmodels.stats.power import TTestIndPower
import pandas as pd
import math
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt


# ### 2. Perform power analysis

# In[3]:


# Perform the power analysis to determine sample size:
analysis = TTestIndPower()

effect = sms.proportion_effectsize(0.50, 0.55)
power = 0.8
alpha = 0.05

result = analysis.solve_power(effect, power = power, nobs1 = None, ratio = 1, alpha = alpha)

print('Sample Size: %.3f' % result)


# ### 3. Import data set

# In[10]:


# Read the data set with Pandas.
df = pd.read_csv("new_bike_shop_AB.csv")

# View the DataFrame.
print(df.shape)
print(df.head())
df.info()


# ### 4. Clean the data

# In[13]:


# Rename the columns.
df_new = df.rename(columns = {'IP Address': 'IPAddress', 'LoggedInFlag': "LoyaltyPage"})

# View the DataFrame.
print(df_new.shape)
print(df_new.head())
df_new.info()


# In[14]:


# Drop duplicate values.
df_new.drop_duplicates(subset ="IPAddress", keep = False, inplace = True)

# Remove unneeded columns.
df_final = df_new.drop(['Unnamed: 0', 'RecordID', 'VisitPageFlag'], axis = 1)

# View the DataFrame.
print(df_final.shape)
df_final.info()


# ### 5. Subset the DataFrame

# In[15]:


# Split data set into ID1 as treatment and ID2 & ID3 as control group.
df_final['Group'] = df_final['ServerID'].map({1: 'Treatment', 2: 'Control', 3: 'Control'})

# View the DataFrame.
print(df_final.shape)
df_final.head()


# In[16]:


# Count the values.
df_final['Group'].value_counts()


# In[17]:


# Create two DataFrames.
control_sample = df_final[df_final['Group'] == 'Control'].sample(n = 1565, random_state = 22)

treatment_sample = df_final[df_final['Group'] == 'Treatment'].sample(n = 1565, random_state = 22)

# View the DataFrames.
print(control_sample)
print(treatment_sample)


# ### 6. Perform A/B testing

# In[18]:


# Perform A/B testing.
# Create variable and merge DataFrames.
ab_test = pd.concat([control_sample, treatment_sample], axis = 0)

ab_test.reset_index(drop = True, inplace = True)

# View the output.
ab_test


# In[19]:


# Calculate the conversion rates:
conversion_rates = ab_test.groupby('Group')['LoyaltyPage']

# Standard deviation of the proportion:
STD_p = lambda x: np.std(x, ddof = 0)
SE_p = lambda x: st.sem(x, ddof = 0)

conversion_rates = conversion_rates.agg([np.mean, STD_p, SE_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']

conversion_rates.style.format('{:.3f}')


# In[22]:


# Calculate the p-value and confidence intervals.
# Import necessary packages.
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

control_results = ab_test[ab_test['Group'] == 'Control']['LoyaltyPage']
treatment_results = ab_test[ab_test['Group'] == 'Treatment']['LoyaltyPage']

n_con = control_results.count()
n_treat = treatment_results.count()

successes = [control_results.sum(), treatment_results.sum()]

nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs = nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs = nobs, alpha = 0.05)

print(f'Z test stat: {z_stat:.2f}')
print(f'P-value: {pval:.3f}')
print(f'Confidence Interval of 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'Confidence Interval of 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# ### 7. Summarise results and explain your answers

# In[24]:


# The change to the homepage slightly decreased the click through to the log in page. 

# The p-value is well over the Alpha value of 0.05, meaning the null hypothesis cannot be rejected. 


# In[ ]:




