#!/usr/bin/env python
# coding: utf-8

# ## Practical_Activity_1.1.11

# ### 1. Prepare your workstation

# In[24]:


# Import necessary libraries.
import pandas as pd
import random
get_ipython().run_line_magic('matplotlib', 'inline')
from dataclasses import dataclass
from sensitivity import SensitivityAnalyzer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ### 2. Specify the model inputs and create a class

# In[25]:


# Create a class and instance
@dataclass
class ModelInputs:
    share_no: int = 500
    buying_price: int = 10
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
    
model_data = ModelInputs()

model_data


# ### 3. Create a function to calculate ROI

# In[27]:


# Create function to calculate ROI:
def roi_per_year(data: ModelInputs):
    year = 0
    prior_roi = 0
    
    while year < 6:
        year = year + 1
        # print(year)
        net_return = (data.selling_price - data.buying_price)         * data.share_no + data.dividend - data.costs
        cost_of_investment = data.buying_price * data.share_no
        
        roi = prior_roi + ((net_return) / (cost_of_investment)) * 100
        print(f'The ROI at year {year} is {roi:,.0f}%.')
        
        prior_roi = roi
        
    return roi

# View the output
roi_per_year(model_data)


# In[34]:


# Create a DataFrame to display the values generated. 
def roi_per_year_df(data: ModelInputs):
    year = 0
    prior_roi = 0
    df_data_tups = []
    
    for year in range(6):
        year = year + 1
        net_return = (data.selling_price - data.buying_price)         * data.share_no + data.dividend - data.costs
        cost_of_investment = data.buying_price * data.share_no
        roi = prior_roi + ((net_return) / (cost_of_investment)) * 100
        
        prior_roi = roi
        # Save the results in a tuple for later building the DataFrame.
        df_data_tups.append((year, roi))
        # Now create the DataFrame.
        df = pd.DataFrame(df_data_tups, columns = ['Year', 'ROI in %'])
        
    return df

# View the output
roi_per_year_df(model_data)


# ### 4. Create a function to show ROI for each year and a single year

# In[35]:


# Function to give annual ROI for each year:
def annualised_roi_per_year(data: ModelInputs):
    year = 0
    prior_roi = 0
    
    for year in range(6):
        year = year + 1
         # print(year)
        net_return = (data.selling_price - data.buying_price)         * data.share_no + data.dividend - data.costs
        cost_of_investment = data.buying_price * data.share_no
        
        roi = prior_roi + ((net_return) / (cost_of_investment)) * 100
        print(f'ROI: {roi}')
        
        annual_roi = ((1 + (roi / 100)) ** (1 / year) - 1) * 100
        print(f'The annualised ROI at year {year} is {annual_roi}%.')
        prior_roi = roi
        
    return annual_roi

# View output.
annualised_roi_per_year(model_data)


# In[37]:


# Function to return annualised ROI for a single year:
def anl_roi_per_year(data: ModelInputs, print_output = True):
    year = 0
    prior_roi = 0
    
    if print_output:
        print(f'Annual ROI over time:')
        
    while year < 10:
        year = year + 1
        net_return = (data.selling_price - data.buying_price)         * data.share_no + data.dividend - data.costs
        cost_of_investment = data.buying_price * data.share_no
        
        # Cost of investment is simply the buying price of each 
        # share multiplied by the number of shares purchased.
        roi = prior_roi + ((net_return) / (cost_of_investment)) * 100
        print(f'ROI: {roi}')
        
        anl_roi = ((1 + (roi / 100)) ** (1 / year) - 1) * 100
        
        return anl_roi
    
# View output.
anl_roi_per_year(model_data)


# ### 5. Run the model based on some changes
# * a.    Initial investment increases by 10% and decreases by 10%.
# * b.    Buying price per share increases by 15% and decreases by 10%.
# * c.    Selling price per share increases by 20% and decreases by 15%.
# * d.    Annual dividend increases by 25% and decreases by 20%. 

# #### a)    Initial investment increases by 10% and decreases by 10%.

# In[38]:


# investment = buying price * share
# NOTE : Assuming that instead of investment, the variable to be used is share no.


# In[39]:


# Buying Share per share increases by 10%.
@dataclass
class ModelInputs:
    share_no : int = 550
    buying_price: int = 11.5
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs
model_data

roi_per_year(model_data)


# In[40]:


# Buying Share per share decreases by 10%.
@dataclass
class ModelInputs:
    share_no : int = 450
    buying_price: int = 11.5
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# #### b) Buying price per share increases by 15% and decreases by 10%

# In[41]:


# Buying price per share increases by 15%.
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 11.5
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# In[42]:


# Buying price per share decreases by 15%.
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 8.5
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# #### c) Selling price per share increases by 20% and decreases by 15%.

# In[43]:


# Selling price per share increases by 20%
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 10
    dividend: int = 500
    selling_price: int = 18
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# In[44]:


# Selling price per share decreases by 20%
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 10
    dividend: int = 500
    selling_price: int = 12
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# #### d) Annual dividend increases by 25% and decreases by 20%.

# In[45]:


# Annual dividend increases by 25%.
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 10
    dividend: int = 625
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# In[46]:


# Annual dividend decreases by 20%.
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 10
    dividend: int = 400
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data

roi_per_year(model_data)


# ### 6. Create a 'good' and 'bad' scenario

# In[47]:


# Create a good bad scenario.
@dataclass
class ModelInputs:
    share_no : int = 500
    buying_price: int = 10
    dividend: int = 500
    selling_price: int = 15
    costs: int = 125
        
model_data = ModelInputs()
model_data


# In[55]:


bad_economy_data = ModelInputs(
    share_no = 100,
    buying_price = 45,
    dividend = 200,
    selling_price = 30,
    costs = 300)

good_economy_data = ModelInputs(
    share_no= 800,
    buying_price= 30,
    dividend= 600,
    selling_price= 55,
    costs= 100)

cases = {
    'Bad': bad_economy_data,
    'Normal': model_data, # Original inputs were set to assume a normal economy
    'Good': good_economy_data}

for case_type, case_inputs in cases.items():
    roi = anl_roi_per_year(case_inputs, print_output=False)
    print(f'Annualised ROI would be {roi} in case of {case_type} economy.')


# ### 7. Perform a what-if scenario

# In[56]:


# Run what-if analysis.
def analyser_what_if(
    share_no  = 500,
    buying_price = 10,
    dividend = 500,
    selling_price = 15,
    costs = 125):
    data = ModelInputs(
        share_no = share_no, 
        buying_price = buying_price, 
        dividend = dividend, 
        selling_price = selling_price,  
        costs = costs)
    
    return annualised_roi_per_year(data)

analyser_what_if()


# In[57]:


# Use Python's 'list comprehensions' syntax to make it easier to adjust the inputs;
# Use i (the iterator) as a temporary variable to store the values's position in the range:
sensitivity_values = {
    'share_no': [i * 100 for i in range(4, 8)],
    'buying_price': [i * 10 for i in range(4, 8)],
    'dividend': [i * 100 for i in range(1, 4)],
    'selling_price': [i * 10 for i in range(10, 25, 5)],
    'costs': [i * 100 for i in range(3, 10)]}

sensitivity_values


# In[60]:


# Run the Python's SensitivityAnalyzer with all the assigned inputs:

sa = SensitivityAnalyzer(
    sensitivity_values,
    analyser_what_if,
    result_name = 'Annual ROI',
    reverse_colors = True,
    grid_size = 3)


# In[62]:


# Display the results using a DataFrame.
styled_dict = sa.styled_dfs(num_fmt = '{:.1f}')


# ### 8. Identify best-case investment scenario

# In[63]:


@dataclass
class ModelInputs:
    share_no : int = 600
    buying_price: int = 20
    dividend: int = 800
    selling_price: int = 22
    costs: int = 100
        
model_data = ModelInputs()
model_data


# In[64]:


def annualised_roi_per_year_for_required_roi(data: ModelInputs):
    year = 0
    prior_roi = 0
    
    for year in range(19):
        year = year + 1        
        net_return = (data.selling_price - data.buying_price)         * data.share_no + data.dividend - data.costs
        cost_of_investment = data.buying_price * data.share_no

        roi = prior_roi + ((net_return) / (cost_of_investment)) * 100 
        annual_roi = ((1 + (roi / 100)) ** (1 / year) - 1) * 100
        
        prior_roi = roi 
    print(f'The annualised ROI at year {year} reaches {annual_roi}% for total shares of {model_data.share_no} with a buying price of {model_data.buying_price}, selling price of {model_data.selling_price}')
    
annualised_roi_per_year_for_required_roi(model_data)


# In[ ]:




