#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# ## Week 1: Predicting business outcomes with regression!

# This week, we will take a deep dive into predictive analytics and, more specifically, regression modelling. You will use this notebook to follow along with the demonstrations throughout the week. 
# 
# This is your notebook. Use it to follow along with the demonstrations, test ideas and explore what is possible. The hands-on experience of writing your own code will accelarate your learning!
# 
# For more tips: https://jupyter-notebook.readthedocs.io/en/latest/ui_components.html

# # 1.1 Advanced Analytics with Python

# ## A/B testing

# ### Conduct Power Analysis

# In[1]:


# [1] Import the 'statsmodel' for statistical calculations and
# [1a] the 'TTestIndPower' class to calculate the parameters for the analysis:
import statsmodels.stats.api as sms
from statsmodels.stats.power import TTestIndPower

# [2] Specify the three required parameters for the power analysis:
# [2a] Specify 'alpha'
alpha = 0.05
# [2b] Specify 'power'
power = 0.8
# [2c] Specify 'effect' and calculate the minimum effect.
effect = sms.proportion_effectsize(0.13, 0.15)

# [3] Perform power analysis by using solve_power() function:
# [3a] Specify an instance of 'TTestIndPower'
analysis = TTestIndPower()
# [3b] Calculate the sample size and list the parameters
result = analysis.solve_power(effect, power = power, nobs1 = None,
                             ratio = 1.0, alpha = alpha)

# [4] Print the output up to 3 decimal places (with lead-in text)
print('Sample Size: %.3f' % result)


# ### Prepare the data in Python

# In[1]:


# Install the relevant modules:
get_ipython().system('pip install scipy')

# Import necessary libraries, packages and classes
import pandas as pd
# Import Python's built-in maths module
import math
import numpy as np
# Import statsmodels stats test and tools
import statsmodels.stats.api as sms
# Import the scipy.stats for more stats functions
import scipy.stats as st
# Import Matplotlib for visualisation tools
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


# Read the 'ab_data.csv' file
df = pd.read_csv("ab_data.csv")

# View the DataFrame
df.head()


# In[3]:


# Check the metadata for information about the file
df.info()


# #### Check for duplicates

# In[4]:


# Use Pandas's duplicated() function to check the user_id column
print(df[df.user_id.duplicated()])


# In[5]:


# Use 'drop_duplicate' to return the Series without the duplicate values
dropped = df.drop_duplicates(subset = "user_id")

dropped.info()


# #### Remove unnecessary columns

# In[6]:


# Use 'dropped.drop' to remove irrelevant columns from the DataFrame
# and specify the column names:
# Specify that 'user_id' and 'timestamp' are columns (i.e. 'axis_1')
final_tab = dropped.drop(["user_id", "timestamp"], axis = 1)

final_tab.head()


# #### Check for errors

# In[7]:


# Use 'crosstab' to compute a simple cross-tabulation between two variables
pd.crosstab(final_tab["group"], final_tab["landing_page"])


# In[9]:


# [1] Specify groups to be dropped
final_tab_cleaned = final_tab[((final_tab.group == 'control') & (final_tab.landing_page == 'old_page')) | ((final_tab.group == 'treatment') & (final_tab.landing_page == 'new_page'))]

# [2] Print the shape of the new 'final' table
print(final_tab_cleaned.shape)
final_tab_cleaned['group'].value_counts()


# In[10]:


# Re-check/compute another simple cross-tabulation
pd.crosstab(final_tab_cleaned['group'], final_tab_cleaned['landing_page'])


# ### Perform random sampling with Pandas

# In[16]:


# Obtain a simple random sample for control and treatment groups with n = 4721;
# set random_stategenerator seed at an arbitrary value of 22.

# Obtain a simple random sample for the control group.
control_sample = final_tab_cleaned[final_tab_cleaned['group'] == 'control'].sample (n = 4721, random_state = 22)
                                   
# Obtain a simple random sample for the treatment group.
treatment_sample = final_tab_cleaned[final_tab_cleaned['group'] == 'treatment'].sample (n = 4721, random_state = 22)


# In[17]:


# [1] Join the two samples.
ab_test = pd.concat([control_sample, treatment_sample], axis = 0)

# [2] Reset the A/B index.
ab_test.reset_index(drop = True, inplace = True)

# [3] Print the sample table.
ab_test


# ### Analyse the data

# #### Calculate basic statistics

# In[19]:


# [1] Import library.
from scipy.stats import sem

# [2] Group the ab_test data set by group and aggregate by converted:
conversion_rates = ab_test.groupby('group')['converted']

# [3] Calculate conversion rates by calculating the means of columns STD_p and SE_p:
conversion_rates = conversion_rates.agg([np.mean, np.std, sem])

# [4] Assign names to the three columns.
conversion_rates.columns = ['conversion_rate', 'std_deviationo', 'std_error']

# [5] Round the output to 3 decimal places
conversion_rates.style.format('{:.3f}')


# #### Calculate statistical significance

# In[22]:


# [1] Import proportions_ztest and proportion_confint from statsmodels:
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# [2] Create a subset of control and treatment results:
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

# [3] Determine the count of the control_results and treatment_result
# sub-datasets and store them in their respective variables:
n_con = control_results.count()
n_treat = treatment_results.count()

# [4a] Create a variable 'success' with the sum of the two data sets
# in a list format:
successes = [control_results.sum(), treatment_results.sum()]

# [4b] Create a variable 'nobs' which stores the values of
# variables n_con and n_treat in list format:
nobs = [n_con, n_treat]

# [5] Use the imported libraries to calculate the statistical values:
z_stat, pval = proportions_ztest(successes, nobs = nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint (successes, nobs = nobs, alpha = 0.05)

# [6a-d] Print the outputs (with lead-in text):
print(f'Z test stat: {z_stat:.2f}')
print(f'P-value: {pval:.3f}')
print(f'Confidence Interval of 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'Confidence Interval of 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')


# ## Conducting a what-if analysis

# ### Set up

# In[1]:


# Install Python Sensitivity Analysis
get_ipython().system('pip install sensitivity')


# In[2]:


# Define classes to contain and encapsulate data.
from dataclasses import dataclass
import pandas as pd
# Import built-in module for generating random numbers.
import random
# Display output inline.
get_ipython().run_line_magic('matplotlib', 'inline')
# Import to replicate a nested loop over the input values.
from sensitivity import SensitivityAnalyzer


# ### Specify inputs

# In[3]:


# Create a DataFrame consisting of various classes using Python's 'dataclass()'
# module and Object Oriented programming (OPP):

@dataclass
class ModelInputs:
    # Define the class and specify the default inputs.
    starting_salary: int = 300000
    promos_every_n_years: int = 5
    cost_of_living_raise: float = 0.025
    promo_raise: float = 0.15
    savings_rate: float = 0.20
    interest_rate: float = 0.07
    desired_cash: int = 1500000
        
# Create an instance of the new class with the default inputs.
model_data = ModelInputs()

# Print the results.
model_data


# ### Calculate wages

# In[4]:


# Get the wage at a given year from the start of the model based on
# the cost of living raises and regular promotions:

def wages_year(data: ModelInputs, year):
    # Every n years we have a promotion, so dividing the years and
    # taking out the decimals gets the number of promotions
    num_promos = int(year / data.promos_every_n_years)
    
    # This is the formula above implemented in Python:
    salary_t = data.starting_salary * (1 + data.cost_of_living_raise)     ** year * (1 + data.promo_raise) ** num_promos
    return salary_t


# In[5]:


# Show the first four salaries in the range and print the
# results using the f-string:

for i in range(4):
    year = i + 1
    salary = wages_year(model_data, year)
    print(f'The wage at year {year} is ${salary:,.0f}.')


# In[6]:


# Change to show the salaries for the first 10 years 
# only and print the results using the f-string:

for i in range(10):
    year = i + 1
    salary = wages_year(model_data, year)
    print(f'The wage at year {year} is ${salary:,.0f}.')


# ### Calculate wealth

# In[7]:


# Calculate the cash saved within a given year by first calculating
# the salary at that year then applying the savings rate:

def cash_saved_during_year(data: ModelInputs, year):
    salary = wages_year(data, year)
    cash_saved = salary * data.savings_rate
    return cash_saved


# In[8]:


# Calculate the accumulated wealth for a given year based on
# previous wealth, the investment rate, and cash saved during the year:

def wealth_year(data: ModelInputs, year, prior_wealth):
    cash_saved = cash_saved_during_year(data, year)
    wealth = prior_wealth * (1 + data.interest_rate) + cash_saved
    return wealth


# In[10]:


# Start with no cash saved.
prior_wealth = 0
for i in range(4):
    year = i + 1
    wealth = wealth_year(model_data, year, prior_wealth)
    print(f'The wealth at year {year} is ${wealth:,.0f}.')


# In[16]:


# Start with no cash saved, view 10 years.
prior_wealth = 0
for i in range(4):
    year = i + 1
    wealth = wealth_year(model_data, year, prior_wealth)
    print(f'The wealth at year {year} is ${wealth:,.0f}.')
    
    # Set next year's prior wealth to this year's wealth.
    prior_wealth = wealth


# ### Calculate retirement

# In[12]:


# Create while loop to run through each year, starting with
# no cash saved:

def years_to_retirement(data: ModelInputs, print_output = True):
    # Start with no cash saved.
    prior_wealth = 0
    wealth = 0
    # The 'year' becomes '1' on the first loop.
    year = 0
    
    if print_output:
        print('Wealths over time:')
    while wealth < data.desired_cash:
        year = year + 1
        wealth = wealth_year(data, year, prior_wealth)
        if print_output:
            print(f'The wealth at year {year} is ${wealth:,.0f}.')
        # Set the next year's prior wealth to this year's wealth.
        prior_wealth = wealth
        
    # Now we have run the while loop, the wealth must be >= desired_cash
    # (whatever last year was set is the years to retirement), which we can print:
    if print_output:
        # \n makes a blank line in the output.
        print(f'\nRetirement:\nIt will take {year} years to retire.')
    return year


# In[14]:


# Using the default inputs, let's see how long it will take
# to reach over $1.5 million in wealth and retire.
years = years_to_retirement(model_data)


# ## Running a sensitivity and scenario analysis

# ## Testing a model's sensitivity with different inputs

# ### Defining functions for calculating the values for sensitivity analysis

# In[18]:


import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)


# In[19]:


# [1] Define the function that accpets the infividual parameters
# rather than the entire data:
def years_to_retirement_separate_args(
    # [2] List the parameters and set their values.
    starting_salary = 60000, promos_every_n_years = 5, cost_of_living_raise = 0.02,
    promo_raise = 0.15, savings_rate = 0.25, interest_rate = 0.05, desired_cash = 1500000):
    # [3] Update the values of the parameters:
    data = ModelInputs(
        starting_salary = starting_salary,
        promos_every_n_years = promos_every_n_years,
        cost_of_living_raise = cost_of_living_raise,
        promo_raise = promo_raise,
        savings_rate = savings_rate,
        interest_rate = interest_rate,
        desired_cash = desired_cash)
    
    return years_to_retirement(data, print_output = False)

# [4] Call the function.
years_to_retirement_separate_args()


# ### Generate random values for the input variables using list comprehensions

# In[20]:


# Use Python's 'list comprehensions' syntax to make it easier to adjust the inputs;
# Use i (the iterator) as a temporary variable to store the values's position in the range:

sensitivity_values = {
    'starting_salary': [i * 10000 for i in range(4, 8)],
    'promos_every_n_years': [i for i in range(4, 8)],
    'cost_of_living_raise': [i / 100 for i in range(1, 4)],
    'promo_raise': [i / 100 for i in range(10, 25, 5)],
    'savings_rate': [i / 100 for i in range(10, 50, 10)],
    'interest_rate': [i / 100 for  i in range(3, 8)],
    'desired_cash': [ i * 100000 for i in range(10, 26, 5)]}


# ### Running the sensitivity analyser module

# In[21]:


# Run the Python's SensitivityAnalyzer with all the assigned inputs:

sa = SensitivityAnalyzer(
    sensitivity_values,
    years_to_retirement_separate_args,
    result_name = 'Years to Retirement',
    reverse_colors = True,
    grid_size = 3)


# ### Display the results

# In[22]:


# Display the results using a DataFrame.
styled_dict = sa.styled_dfs(num_fmt = '{:.1f}')


# ## Scenario analysis

# In[23]:


# The function to calculate 'bad' economy:
bad_economy_data = ModelInputs(
    starting_salary = 100000,
    promos_every_n_years = 8,
    cost_of_living_raise = 0.01,
    promo_raise = 0.07,
    savings_rate = 0.15,
    interest_rate = 0.03)

# The function to calculate 'good' economy:
good_economy_data = ModelInputs(
    starting_salary = 500000,
    promos_every_n_years = 4,
    cost_of_living_raise = 0.03,
    promo_raise = 0.20,
    savings_rate = 0.35,
    interest_rate = 0.06)

cases = {
    'Bad': bad_economy_data,
    'Normal': model_data, # Original inputs were set to assume a 'normal' economy
    'Good': good_economy_data}


# In[24]:


# Run the model with the three scenarios and print the results (with a text string):
for case_type, case_inputs in cases.items():
    ytr = years_to_retirement(case_inputs, print_output = False)
    print(f'It would take {ytr} years tp retire in a {case_type} economy.')


# ## Assigning probabilities

# In[25]:


# Note: These values are arbitrary (i.e. they have been randomly allocated)
# and are only used for demonstration.

case_probabilities = {
    'Bad': 0.2,
    'Normal': 0.5,
    'Good': 0.3}


# In[27]:


# Run the model by taking the expected value over the three cases;
# Print the results with a text string:

expected_ytr = 0
for case_type, case_inputs in cases.items():
    ytr = years_to_retirement(case_inputs, print_output = False)
    weighted_ytr = ytr * case_probabilities[case_type]
    expected_ytr += weighted_ytr
    
    print(f'It would take {expected_ytr:.0f} years to retire given a {case_probabilities["Bad"]:.0%} change of a bad economy and {case_probabilities["Good"]:.0%} change of a good economy.')


# In[ ]:





# # 

# # 1.2 Linear regression using Python

# ## Simple linear regression analysis

# ### 1. Find the line of best fit

# In[1]:


# Import Numpy for statistical calculations and Matplotlib for plotting functions:
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Regression analysis function:
def estimate_coef(x, y):
    n = np.size(x) # Specify the size or number of points.
    
    # Calculate the mean of x and y:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate the cross_deviation and deviation around x:
    SS_xy = np.sum(y * x) - n * mean_y * mean_x
    SS_xx = np.sum(x * x) - n * mean_x * mean_x
    
    # Calculate the regression coefficients:
    m = SS_xy / SS_xx
    b = mean_y - m * mean_x
    
    return (b, m)


# In[3]:


def plot_regression_line(x, y, b):
    # [1] Use scatterplot to plot the actual points:
    plt.scatter(x, y, color = "g", marker = "o", s = 30)
    
    # [2] Set the predicted response vector using the linear equation:
    y_pred = b[0] + b[1] * x
    
    # [3] Plot the regression line (in red):
    plt.plot(x, y_pred, color = "r")
    
    # [4] Add two labels for clarity:
    plt.xlabel('x')
    plt.ylabel('y')
    
    # [5] Set a function to display the plot:
    plt.show()


# ### 2. Add data

# In[6]:


def main():
    # [1] Enter small data set (in this case) manually as an array:
    x = np.array([0.9, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5])
    y = np.array([20000, 22000, 23500, 26000, 25000, 28250, 29300,
                  33000, 34255, 45000])
    
    # [2] Calculate the coefficients:
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nc = {}            \nm = {}".format(b[0], b[1]))
    
    # [3] Plot the regression line (i.e. y = mc + c):
    plot_regression_line(x, y, b)
    
main()


# ### 3. Simplify functions and calculations with NumPy

# In[8]:


# [1] Re-enter the values of x and y in your Notebook:
x = np.array([0.9, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.5])
y = np.array([20000, 22000, 23500, 26000, 25000, 28250, 29300, 33000, 34255, 45000])

# [2] Create a basic scatterplot.
plt.plot(x, y, 'o')

# [3] Obtain m (slope) and c (intercept) of the linear regression line:
m, c = np.polyfit(x, y, 1)

# [4] Add the linear regression line to the scatterplot:
plt.plot(x, m * x + c)


# ## The OLD method and the statsmodels package

# ### 1. Practice the OLD method using the statsmodels package

# In[2]:


# Import the 'statsmodels' package along with Numpy, Pandas and Matplotlib:
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Import and read the data file:
df_test = pd.read_csv("test.csv")

df_test


# In[4]:


# Define the dependent variable.
y = df_test['y']

# Define the independent variable.
x = df_test['x']


# In[5]:


# Check for linearity with scatterplot.
plt.scatter(x, y)


# In[6]:


# Create formula and pass through OLS method:
f = 'y ~ x'
test = ols(f, data = df_test).fit()


# In[7]:


# Print the regression table.
test.summary()


# ### 2. Print useful values

# In[8]:


# Extract the estimated parameters.
print("Parameters: ", test.params)
# Extract the standard errors.
print("Standard errors: ", test.bse)
# Extract the predicted values.
print("Predicted values: ", test.predict())


# ### 3. Create a linear equation and plotting regression

# In[9]:


# Set the x coefficient to '1.0143' and the constant to '-0.4618'
# to generate the regression table:

y_pred = 1.0143 * df_test['x'] - 0.4618

y_pred


# In[10]:


# [1] Plot the data points.
plt.scatter(x, y)

# [2] Plot the regression line (in red).
plt.plot(x, y_pred, color = 'red')

# [3] Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

plt.show()


# ## Linear regression with scikit-learn

# ### Install and impor the required modules and packages

# In[13]:


# Install the necessary modules.
get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install statsmodels')
get_ipython().system('pip install scipy')
get_ipython().system('pip install matplotlib')


# In[14]:


# Import library for statistical analysis.
import statsmodels.api as sm
# Import inbuilt data sets in sklearn library.
from sklearn import datasets
# Import for numerical calculations.
import numpy as np
# Import sklearn's linear model algorithm.
from sklearn import linear_model
# Import Pandas library.
import pandas as pd
# Import for plot generation.
import matplotlib.pyplot as plt

# Import metrics for measuring linear model fit:
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# ### Create a DataFrame

# In[15]:


# Read the 'salary_data.csv' file.
data = pd.read_csv("salary_data.csv")

# Print the table.
data.head()


# In[16]:


# More info about the DataFrame.
data.info()


# ### Determine basic statistics

# In[17]:


# Print descriptive statistics.
data.describe()


# ### Run the regression

# In[18]:


# Define the dependent variable.
y = data['Salary'].values.reshape(-1, 1)

# Define the independent variable.
x = data['YearsExperience'].values.reshape(-1, 1)


# In[19]:


lm = LinearRegression()

# Fit the model.
lm.fit(x, y)


# ### Make predictions

# In[20]:


# Print the target values.
lm.predict(y)


# In[21]:


## Call the intercept.
lm.intercept_


# ### Estimate coefficients

# In[22]:


# Estimate coefficients.
lm.coef_


# In[33]:


# [1] Create a scatterplot (with red data points).
plt.scatter(x, y, color = 'red')
# [2] Create a regression line in green.
plt.plot(x, lm.predict(x), color = 'green')
# [3] Set the title for the graph.
plt.title("Years of Experience vs Avg. Salary")
# [4] Set the label for the x-axis.
plt.xlabel("Years of Experience")
# [5] Set the label for the y-axis.
plt.ylabel("Avg. Salary")
# [6] Print the graph.
plt.show()


# ### Apply the linear regression model

# In[25]:


# Use the predict() method with an array to call the 
# salaries for each number of years' experience.
predictedSalary = lm.predict([[5], [10], [15], [20], [25], [30]])

# Print the results.
print(predictedSalary)


# ### Run the regression on subsets

# In[30]:


# [1] Import the sklearn module.
from sklearn.model_selection import train_test_split

# [2] Create the subset (50/50).
# Control the shuffling/avoid variation in values between variable.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.5,
                                                   random_state = 100)


# ### Run a regression test and visualise the training data

# In[31]:


# Linear regression; fitting the model.
lm.fit(x_train, y_train)


# In[32]:


# Predict the training set values.
y_pred = lm.predict(x_train)


# ### Plot and visualise the training data

# In[34]:


# [1] Create a scatterplot (with red data points).
plt.scatter(x_train, y_train, color = 'red')
# [2] Create a regression line in green.
plt.plot(x_train, y_pred, color = 'green')
# [3] Set the title for the graph.
plt.title("Years of Experience vs Avg. Salary")
# [4] set the label for the x-axis.
plt.xlabel("Years of Experience")
# [5] Set the label for the y-axis.
plt.ylabel("Avg. Salary")
# [6] Print the graph.
plt.show()


# In[35]:


# Print R-squared value of the training data.
print(lm.score(x_train, y_train))


# ### Calculate intercept and coefficient values

# In[37]:


# Print the intercept value.
print("Intercept value: ", lm.intercept_)
# Print the coefficient value.
print("Coefficient value: ", lm.coef_)


# ## Testing the model

# ### Run regression and visualise test data

# In[38]:


# Linear regression; fitting the model.
lm.fit(x_test, y_test)


# ### Predict test set values

# In[39]:


# Predict the test set values.
y_pred = lm.predict(x_test)


# ### Visualise

# In[40]:


# [1] Create a scatterplot (with red data points).
plt.scatter(x_test, y_test, color = 'red')
# [2] Create a regression line in green.
plt.plot(x_test, y_pred, color = 'green')
# [3] Set the title for the graph.
plt.title("Years of Experience vs Avg. Salary")
# [4] set the label for the x-axis.
plt.xlabel("Years of Experience")
# [5] Set the label for the y-axis.
plt.ylabel("Avg. Salary")
# [6] Print the graph.
plt.show()


# ### Print R-squared value

# In[42]:


# Print R-squared value of the test data.
print(lm.score(x_test, y_test))


# ### Print intercept and coefficient values

# In[43]:


# Print the intercept value.
print("Intercept value: ", lm.intercept_)
# Print the coefficient value.
print("Coefficient value: ", lm.coef_)


# ## Using statsmodels to check homoscedasticity

# In[45]:


# Import libraries.
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms


# In[46]:


# Import the test data set and run the OLS on the data:
df_test = pd.read_csv("test.csv")

f = 'y ~ x'
test = ols(f, data = df_test).fit()


# In[47]:


# Run the Breusch-Pagan test function on the model residuals and x-variables:
test = sms.het_breuschpagan(test.resid, test.model.exog)


# In[48]:


# Print the results of the Breusch-Pagan test:
terms = ['LM stat', 'LM Test p-value', 'F-stat', 'F-test p-value']
print(dict(zip(terms, test)))


# In[ ]:





# # 1.3 CAPM and multiple regression

# ## Multiple linear regression

# In[49]:


# Import libraries.
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
from sklearn import linear_model
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[50]:


# Import csv file.
hp = pd.read_csv("house_prices.csv")

# Print the DataFrame.
hp.head()


# In[51]:


# Print more information about the DataFrame.
hp.info()


# In[53]:


# Define the variables.
y = hp['Value']
X = hp[['Rooms', 'Distance']]


# In[54]:


# Fit the regression model.
mlr = linear_model.LinearRegression()

mlr.fit(X, y)

# Call the predictions for X (array).
mlr.predict(X)


# In[55]:


# Print the R-squared value.
print("R-squared: ", mlr.score(X, y))
# Print the intercept.
print("Intercept: ", mlr.intercept_)
# Print the coefficient.
print("Coefficient: ")
# Map a similar index of multiple containers (to be used as a single entity).
list(zip(X, mlr.coef_))


# In[57]:


# Create a new variable 'New_Rooms' and define it as 5.75.
New_Rooms = 5.75
# Create a new variable 'New_Distance' and define it as 15.2.
New_Distance = 15.2

print('Predicted Value: \n', mlr.predict([[New_Rooms, New_Distance]]))


# In[58]:


# Create a new variable 'New_Rooms' and define it as 6.75.
New_Rooms = 6.75
# Create a new variable 'New_Distance' and define it as 15.2.
New_Distance = 15.2

print('Predicted Value: \n', mlr.predict([[New_Rooms, New_Distance]]))


# In[60]:


# Split the data in 'train' (80%) and 'test' (20%) subsets:
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split( X, y, test_size = 0.2, random_state = 5)


# In[64]:


# Training the model using the 'statsmodels' OLS library:
# Fit the model with the added constant.
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

# Set the predicted response vector.
y_pred = model.predict(sm.add_constant(X_test))

# Call a summary of the model.
print_model = model.summary()

# Print the summary.
print(print_model)


# In[65]:


# Run model on 'test' subset.
mlr = LinearRegression()

# Fit the model.
mlr.fit(X_test, y_test)

# Call the predictions for X in the test subset.
y_pred_mlr = mlr.predict(X_test)

# Print the predictions.
print("Prediction for test subset: {}".format(y_pred_mlr))


# In[66]:


# Print the R-squared value.
print(mlr.score(X_test, y_test) * 100)


# ## Check for multicollinearity

# In[70]:


# Add a constant.
x_temp = sm.add_constant(X_train)

# Create an empty DataFrame.
vif = pd.DataFrame()

# Calculate the VIF (variance inflation factor).
vif['VIF Factor'] = [variance_inflation_factor(x_temp.values, i) for i in range(x_temp.values.shape[1])]

# Create the feature column.
vif['feature'] = x_temp.columns

# Print the values to two decimal places.
print(vif.round(2))


# In[71]:


# Call the 'metrics.mean_absolute_error' function.
print('Mean Absolute Error (Final): ', metrics.mean_absolute_error(y_test, y_pred))

# Call the 'metrics.mean_squared_error' function.
print('Mean Squared Error (Final): ', metrics.mean_squared_error(y_test, y_pred))


# In[ ]:




