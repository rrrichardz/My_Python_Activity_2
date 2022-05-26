#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# ## Week 2: Predicting outcomes using classification and clustering

# This week we will continue to learn about more different types of machine learning algorithms (classification and clustering) used in predictive analytics to analyse business trends and patterns to obtain meaningful business insights. We will also learn about the available Python tools that enable analysts to automate these tasks. Python libraries, such as Pandas, Matplotlib and Seaborn, can provide the tools to help businesses predict trends and make decisions. 
# 
# In addition, we will learn how to use these Python tools to create and test classification models, which are useful for analysing categorical data such as product types and customer segments. We will then continue to build on our knowledge of regression analysis techniques by learning how to use binary logistical regression, multinomial logistical regression, and support vector machines.
# 
# This is your notebook. Use it to follow along with the demonstrations, test ideas and explore what is possible. The hands-on experience of writing your own code will accelarate your learning!
# 
# For more tips: https://jupyter-notebook.readthedocs.io/en/latest/ui_components.html

# # 2.1 Classification with Python

# ## Binary logistic regression (BLR)

# ### 1. Import and read the data set

# In[1]:


# Import all necessary packages.
import numpy as np
import pandas as pd

# Read the data file.
df = pd.read_csv("Customer_data.csv")

# Print the DataFrame.
df.head()


# ### 2. Determine the data types of each column

# In[2]:


# Find the data types of columns (e.g. replace strings with single
# words or round numeric values to a specific number of decimals).
df.dtypes


# ### 3. Determine the shape of the data set

# In[3]:


# Determine the shape of the data set (one of the assumptions to 
# be met for logistic regression is a large data set).
df.shape


# ### 4. Check for missing values

# In[4]:


# Determine missing values, column names, shape of data set, and data type:
df.info()


# ### 4. Determine object containing counts of unique values

# In[5]:


# Specify the DataFrame and column & add/determine the values.
df['Edu'].value_counts()


# ### 5. Update the categories in the 'Edu' column

# In[6]:


# Specify the DataFrame and column name,
# Specify the DataFrame and the column name that contains
# the string to be changed
# Specify the word to be changed that the string contains
# and the new name:

df.loc[df['Edu'].str.contains('basic'), 'Edu'] = 'pre-school'
df.loc[df['Edu'].str.contains('university'), 'Edu'] = 'uni'
df.loc[df['Edu'].str.contains('high'), 'Edu'] = 'high-school'
df.loc[df['Edu'].str.contains('professional'), 'Edu'] = 'masters'
df.loc[df['Edu'].str.contains('illiterate'), 'Edu'] = 'other'
df.loc[df['Edu'].str.contains('unknown'), 'Edu'] = 'other'

# Display all the unique values/check changes.
df['Edu'].unique()


# In[7]:


df['Edu'].value_counts()


# ### 6. Create dummy variables

# In[8]:


# [1] Name new DataFrame and convert categorical variables to dummy variables:
cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# [2] Use the for loop keyword to specify what actions to apply to
# all the 'var' items:
# [2a] Specify what needs to apply to all the variables:
for var in cat_vars:
    cat_list = 'var' + '_' + var
    # [2b] Specify details of the categorical list.
    cat_list = pd.get_dummies(df[var], prefix = var)
    # [2c, 3] Indicate the joining of the DataFrames.
    cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan',
               'Comm', 'Month', 'DOW', 'Last_out']
# [4] Set a temporary DataFrame and add values.
df_vars = df.columns.values.tolist()
# [5] Indicate what columns are kept.
to_keep = [i for i in df_vars if i not in cat_vars]

# [6] Define new DataFrame.
df_fin = df[to_keep]

# [7] Print the column.
df_fin.columns.values


# ### 7. Balance the data

# In[9]:


# Determine if values in a column are balanced.
df['Target'].value_counts()


# In[10]:


# Handles unbalanced data (scikit-learn needed).
get_ipython().system('pip install imblearn')
# Optimised linear, algebra and integrations (scientific).
get_ipython().system('pip install scipy')
# Simple tools for predictive data analytics.
get_ipython().system('pip install scikit-learn')
# Oversampling technique; creates new samples from data.
get_ipython().system('pip install SMOTE')


# In[11]:


# [1] Import all the ncessary packages:
# [1a] Assists with providing classes and functions
# to estimate many different statistical methods.
import statsmodels.api as sm
import imblearn
# [1b] Helps split data into sets to create BLR.
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# [1c] Indicates situations that aren't necassarily exceptions.
import warnings
warnings.filterwarnings("ignore")


# In[8]:


# [2] Create the DataFrame to use as df_fin and replace
# missing values with zero.
df_fin = df_fin.fillna(0)

# [3]Specify the variables:
X = df_fin.loc[:, df_fin.columns != 'Target']
y = df_fin.loc[:, df_fin.columns == 'Target']

# [4] Create a new DataFrame and
# [4a] Apply SMOTE function as the target variable is not balanced.
os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state = 0)
# [5] Specify column values.
columns = X_train.columns
# [6] Specify the new data sets.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)

# [7] Create two DataFrames for X and y:
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)

os_data_y = pd.DataFrame(data = os_data_y, columns = ['Target'])

# [8] Print the DataFrame.
print('Length of oversampled data is', len(os_data_X))
os_data_y


# In[13]:


# Determine if values in a column are balanced by counting the values:
os_data_y['Target'].value_counts()


# ### Apply VIF to check multicollinearity

# In[14]:


# [1] Select all the numeric columns.
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# [2] Create a DataFrame to contain the numeric columns.
df_num = df.select_dtypes(include = numerics)

# View the DataFrame.
print(df_num.head())

# [3] Import the VIF package.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# [4] Create a VIF DataFrame.
vif_data = pd.DataFrame()
vif_data['feature'] = df_num.columns

# [5] Calculate the VIF for each feature.
vif_data['VIF'] = [variance_inflation_factor(df_num.values, i)
                  for i in range(len(df_num.columns))]

# [6] View the output.
vif_data


# In[15]:


# Test correlations.
_correlations = df_num.corr()

# View output.
_correlations


# In[16]:


# Visualise the correlation.
import seaborn as sns

# Set fig and font size.
sns.set(rc = {'figure.figsize': (15, 8)})
sns.set(font_scale = 1.5)

# Plot heatmap.
dataplot = sns.heatmap(_correlations, cmap = 'YlGnBu', annot = True)


# ### Box-Tidwell test

# In[17]:


# [1] Import necessary libraries, modules, classes and packages.
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# [2] Select all the continuous variables.
continuous_vars = list(df_num.columns[:-1])

# [3] Make a copy of the DataFrame.
df_test = df.copy()

# [4] Add logit transform interaction terms (natural log) for
# continuous variables e.g.. Age * Log(Age).
for var in continuous_vars:
    df_test[f'{var}:Log_{var}'] = df_test[var].apply(lambda x: x * np.log(x))
    
    # [5] Keep columns related to continuous variables.
    cols_to_keep = continuous_vars + [_ for _ in df_test.columns if 'Log_' in _]
    
# [6] View output.
list(cols_to_keep)


# In[18]:


# Redefining variables to include interaction terms
# [1] Replace missing values with 0
X_lt = df_test[cols_to_keep].fillna(0)

# [2] Add constant term.
X_lt_constant = sm.add_constant(X_lt, prepend = False)

# [3] Building the model and fit the data (using statsmodel's Logit)
logit_results = GLM(y, X_lt_constant, family = families.Binomial()).fit()

# [4] Display summary results
print(logit_results.summary())


# In[19]:


# [1] Use the model you created.
logit_results = GLM(y, X_lt, family = families.Binomial()).fit()
predicted = logit_results.predict(X_lt)

# [2] Getting log odds values
log_odds = np.log(predicted / (1 - predicted))

# [3] View output.
log_odds


# In[20]:


import matplotlib.pyplot as plt

def vis_var(varname):
    # Visualise predictor variable vs logit values for Age.
    plt.scatter(x = X_lt[varname].values, y = log_odds)
    plt.xlabel(varname)
    plt.ylabel('Log-odds')
    plt.show()
    
# Look at one that is not libearly related to the log odds.
vis_var('Age')


# In[21]:


# Look at one that is linearly related to the log oods.
vis_var('Month_rate')


# In[22]:


vis_var('Duration')


# ### Recursive feature elimination (RFE)

# In[9]:


# Resursive feature elimination:
# [1] Create a new DataFrame.
data_final_vars = df_fin.columns.values.tolist()

# [2] Set the variables:
y = ['Target']
X = [i for i in data_final_vars if i not in y]

# [3] Import two packages from sklearn:
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# [4] Indicate 'logreg' equals 'LogisticRegression()'.
logreg = LogisticRegression()

# [5] Specify 'rfe' value and no. of features.
rfe = RFE(logreg)

# [6] Indicate the fit with 'fit()'.
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())

# [7] Print the two rfes:
print(rfe.support_)
print(rfe.ranking_)


# #### Selecting necessary columns for BLR

# In[13]:


# [1] Name the new DataFrame and [2] specify all the columns for BLR:
nec_cols = ['Status_divorced', 'Status_married', 'Status_single',
            'Status_unknown', 'Edu_high-school', 'Edu_masters', 
            'Edu_other', 'Edu_pre-school', 'Edu_uni', 'House_no',
            'House_unknown', 'House_yes', 'Loan_no', 'Loan_unknown',
            'Loan_yes', 'DOW_fri', 'DOW_mon']

# [3a] Set the independent variable.
X = os_data_X[nec_cols]

# [3b] Set the dependent variable.
y = os_data_y['Target']

# [4] Set the logit() to accept y and x as parameters and return the logit object:
logit_model = sm.Logit(y, X)

# [5] Indicate result = logit_model.fit() function.
result = logit_model.fit()

# [6] Print the results.
print(result.summary2())


# ### Checking BLR accuracy

# In[14]:


# [1] Import neceassary packages:
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# [2] Split X and y data sets into 'train' and 'test' in a 30 : 70 ratio:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                   random_state = 0)

# [2a] Set LogisticRegression() to 'logreg'.
logreg = LogisticRegression()

# [2b] Fit the X_train and y_train data sets to logreg.
logreg.fit(X_train, y_train)


# In[15]:


# Determine BLR model’s accuracy:
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'      .format(logreg.score(X_test, y_test)))


# ### Using confusion matrix to determine classification accuracy

# In[16]:


# Create the confusion matrix to test classification accuracy in BLR:
# [1] Import the necessary package to create the confusion matrix.
from sklearn.metrics import confusion_matrix

# [2] Indicate the confusion matrix needs to be created.
confusion_matrix = confusion_matrix(y_test, y_pred)

# [3] Print the confusion matrix.
print(confusion_matrix)


# In[17]:


# [1] Import the necessary package.
from sklearn.metrics import classification_report

# [2] Print a report on the model's accuracy.
print(classification_report(y_test, y_pred))


# ## Multinomial logistic regression (MLR)

# ### 1. Import and read the data set

# In[37]:


# Import all the necessary packages: Pandas, NumPy, SciPy, Sklearn, StatsModels:
import pandas as pd
import numpy as np
import scipy as scp
import sklearn
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[38]:


# Upload the CSV file.
oysters = pd.read_csv("oysters.csv")

# Print the columns.
oysters.columns


# In[39]:


# View the DataFrame.
oysters.info()


# In[40]:


# Apply the value_counts() method, and
# Assign the results to a new DataFrame:
oysters_sex = oysters['sex'].value_counts()

# Print the contents.
print(oysters_sex)


# ### 2. Identifying the dependent variable column

# In[41]:


# [1] Set the independent and dependent variables:
X = oysters.drop(['sex'], axis = 1)
y = oysters['sex']

# [2] Print to check 'sex' column was dropped.
print(list(X.columns.values))


# In[42]:


# [3] Specify the train and test subsets and
# [3a] Use 30% as the 'test_size' and a random_state of one:
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.30, random_state = 1, stratify = y)

# [4] Print the shape of all the train and tes sets:
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### 3. Build the model

# In[43]:


# [1] Import the MinMaxScaler to normalise the data.
from sklearn.preprocessing import MinMaxScaler

# [2] Create a function and [2a] set values.
scaler = MinMaxScaler(feature_range = (0, 1))

# [3] Add the X_train data set to the 'scaler' function:
scaler.fit(X_train)

# [3a] Specify X_train data set.
X_train = scaler.transform(X_train)

# [3b] Specify X_test data set.
X_test = scaler.transform(X_test)


# In[44]:


# [1] Define the MLR model and [1a] set predictions and parameters:
MLR = LogisticRegression(random_state = 0, multi_class = 'multinomial',
                        penalty = 'none', solver = 'newton-cg').fit(X_train, y_train)

# [2] Set the predictions equal to the 'MLR' function and
# [2a] specify the DataFrame.
preds = MLR.predict(X_test)

# [3] Set the parameters equal to the DataFrame and
# [3a] add the 'get_params' function.
params = MLR.get_params()

# [4] Print the parameters.
print(params)


# In[45]:


# Evaluate the MLR intercept and coefficients.
print('Intercept: \n', MLR.intercept_)
print('Coefficients: \n', MLR.coef_)


# ### 4. Create a linear equation from the logit model

# In[46]:


# [1] Name the model and [2] set model to the function:
logit_model = sm.MNLogit(y_train, sm.add_constant(X_train))

logit_model

# [3] Specify how the function returns the results.
result = logit_model.fit()

# [4] Print the reprot as a 'result.summary()' function.
print('Summary for Sex: I/M :\n', result.summary())


# ### 5. Check the accuracy of the model

# In[47]:


# Create and print a confusion matrix:

# 'y_test' as the first argument and the predictions as the second argument:
confusion_matrix(y_test, preds)

# Transform confusion matrix into an array:
cmatrix = np.array(confusion_matrix(y_test, preds))

# Create the DataFrame from cmatrix array
pd.DataFrame(cmatrix, index = ['female', 'infant', 'male'],
            columns = ['predicted_female', 'predicted_infant',
                      'predicted_male'])


# In[48]:


# Determine accuracy statistics:
print('Accuray score:', metrics.accuracy_score(y_test, preds))

# Create classification report:
class_report = classification_report(y_test, preds)

print(class_report)


# ### 6. Visualise the MLR model

# In[50]:


# [1] Improt matplotlib to create a visualisation.
import matplotlib.pyplot as plt

# [2] Define confusion matrix.
cm = confusion_matrix(y_test, preds)

# [3] Create visualisation for the MLR:
fig, ax = plt.subplots(figsize = (10, 10))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks = (0, 1, 2), ticklabels = ('predicted_female',
                                             'predicted_infant',
                                             'predicted_male'))
ax.yaxis.set(ticks = (0, 1, 2), ticklabels = ('female', 'infant', 'male'))

for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha = 'center', va = 'center', color = 'red')
        
plt.show()


# ## Building an support vector machine model (SVM)

# ### 1. Import and read the data set

# In[1]:


# [1] Import all the nceassary packages:
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
# Note: Indicates situations that aren't necessarily exceptions.
import warnings
warnings.filterwarnings("ignore")

# [2] Read the date file with Pandas.
df = pd.read_csv("Customer_data.csv")

# [3] Sense-check the data.
df.info()


# ### 2. Clean the data

# In[2]:


# Update all the details of the education column:
df.loc[df['Edu'].str.contains('basic'), 'Edu'] = 'pre-school'
df.loc[df['Edu'].str.contains('university'), 'Edu'] = 'uni'
df.loc[df['Edu'].str.contains('high'), 'Edu'] = 'high-school'
df.loc[df['Edu'].str.contains('professional'), 'Edu'] = 'masters'
df.loc[df['Edu'].str.contains('illiterate'), 'Edu'] = 'other'
df.loc[df['Edu'].str.contains('unknown'), 'Edu'] = 'other'

# Display all the unique values/check changes.
df['Edu'].unique()


# ### 3. Create dummy variables

# In[4]:


# Convert categorical variables to dummy variables:
cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# Specify what needs to apply to all the variabels
for var in cat_vars:
    # Specify details of the categorical list
    cat_list = pd.get_dummies(df[var], prefix = var)
    # Indicate the joining of the DataFrames
    df1 = df.join(cat_list)
    # Set the old DataFrame with new df with dummy values
    df = df1
    
    cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# Set the temporary DataFrame and add values
df_vars = df.columns.values.tolist()
# Indicate what columns are kept
to_keep = [i for i in df_vars if i not in cat_vars]

# Define a new DataFrame
df_fin = df[to_keep]

# Print the column
df_fin.columns.values


# ### 4. Balance the data

# In[5]:


# [1] Create the DataFrame to use as 'df_fin' and replace missing values with 0:
df_fin = df_fin.fillna(0)


# In[19]:


# [2] Specify only the necessary columns for BLR:
nec_cols = ['Status_divorced', 'Status_married',
            'Status_single', 'Status_unknown', 
            'Edu_high-school', 'Edu_masters', 
            'Edu_other', 'Edu_pre-school', 
            'Edu_uni', 'House_no', 'House_unknown',
            'House_yes', 'Loan_no', 'Loan_unknown', 
            'Loan_yes', 'DOW_fri', 'DOW_mon']

# [3a] Set the independent variable.
X = df_fin[nec_cols]

# [3b] Set the dependent variable.
y = df_fin.loc[:, df_fin.columns == 'Target']

# [4] Create a new DataFrame and
# [4a] apply SMOTE as the target variable is not balanced.
os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 0)

# [5] Specify column values.
columns = X_train.columns

# [6] Specify the new data sets.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)

# [7] Create two DataFrames, one for X and one for y:
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y = pd.DataFrame(data = os_data_y, columns = ['Target'])

# [8] Print the DataFrame.
print('Length of oversampled data is ', len(os_data_X))

os_data_y


# In[20]:


# Determine if values in a column are balanced.
os_data_y['Target'].value_counts()


# ### 5. Build and apply the SVM

# In[22]:


# [1] Import the 'svm' package from sklearn.
from sklearn import svm
# [2] Import the 'confusion_matrix' package.
from sklearn.metrics import confusion_matrix

# [3] Create an svm classifier using [3a] a linear kernel.
clf = svm.SVC(kernel = 'linear', gamma = 'scale')

# [4] train the model using the training sets.
clf.fit(os_data_X, os_data_y)

# [5] Predict the response for the test data set.
y_pred = clf.predict(X_test)


# ### 6. Check the accuracy of the model

# In[23]:


# [1] Import the scikit-learn metrics module for an accuracy calculation:
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# Print the confusion matrix.
print(confusion_matrix(y_test, y_pred))

# [3a] Specify model accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

# [3b] Specify model precision: what percentage of
# positive tuples are labelled as such?
print('Precision:', metrics.precision_score(y_test, y_pred))

# [3c] Specify model recall: how good is the model at
# correctly predicting positive classes?
print('Recall:', metrics.recall_score(y_test, y_pred))


# In[ ]:





# # 

# # 2.2 Decision tress with Python

# ## Classification decision trees

# ### 1. Import libraries and read data file

# In[2]:


# [1] Import all necessary libraries:
import pandas as pd
import numpy as np
import scipy as scp
import sklearn
from sklearn import metrics
# Note: Provides classes and functions to estimate many different statistical methods.
import statsmodels.api as sm

# Note: Helps split data into sets to create BLR.
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Note: Indicates situations that aren't neceassarily exceptions.
import warnings
# [1a] Filter out any warning messages.
warnings.filterwarnings("ignore")

# [2] Read the provided CSV file.
df = pd.read_csv("Customer_data.csv")

# [3] Print a summary of the DataFrame to sense-check it.
df.info()


# ### 2. Update the variables in the 'Edu' column

# In[3]:


# [1] Update all the details of the education column:
df.loc[df['Edu'].str.contains('basic'), 'Edu'] = 'pre-school'
df.loc[df['Edu'].str.contains('university'), 'Edu'] = 'uni'
df.loc[df['Edu'].str.contains('high'), 'Edu'] = 'high-school'
df.loc[df['Edu'].str.contains('professional'), 'Edu'] = 'masters'
df.loc[df['Edu'].str.contains('illiterate'), 'Edu'] = 'other'
df.loc[df['Edu'].str.contains('unknown'), 'Edu'] = 'other'

# [2] Display all the unique values/check changes.
df['Edu'].unique() 


# ### 3. Create dummy variables

# In[4]:


# Convert categorical variables to dummy variables:
cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# Specify what needs to apply to all the variabels
for var in cat_vars:
    # Specify details of the categorical list
    cat_list = pd.get_dummies(df[var], prefix = var)
    # Indicate the joining of the DataFrames
    df1 = df.join(cat_list)
    # Set the old DataFrame with new df with dummy values
    df = df1
    
    cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# Set the temporary DataFrame and add values
df_vars = df.columns.values.tolist()
# Indicate what columns are kept
to_keep = [i for i in df_vars if i not in cat_vars]

# Define a new DataFrame
df_fin = df[to_keep]

# Print the column
df_fin.columns.values


# ### 4. Balance the data

# In[5]:


# [1] Create a DataFrame to use as df_fin and replace missing valeus with 0.
df_fin = df_fin.fillna(0)

# [2] Select necessary columns:
nec_cols = [ 'Status_divorced', 'Status_married',
            'Status_single', 'Status_unknown', 
            'Edu_high-school', 'Edu_masters', 
            'Edu_other', 'Edu_pre-school', 
            'Edu_uni', 'House_no', 'House_unknown',
            'House_yes', 'Loan_no', 'Loan_unknown', 
            'Loan_yes', 'DOW_fri', 'DOW_mon']

X = df_fin[nec_cols]
y = df_fin['Target']

# [3] Create a new DataFrame and
# [3a] apply SMOTE as the target variable is not balanced.
os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 0)

# [4] Specify column values.
columns = X_train.columns
# [5] Specify the new data sets.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)

# [6] Create two DataFrames for X and y:
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y = pd.DataFrame(data = os_data_y, columns = ['Target'])

# [7] Print/check the DataFrame:
print('Length of oversampled data is ', len(os_data_X))

os_data_y


# In[6]:


# Determine if values in a column is balanced.
os_data_y['Target'].value_counts()


# ### 5. Build and apply the decision tree model

# In[7]:


# [1] Import the 'DecisionTreeClassifier' class from sklearn.
from sklearn.tree import DecisionTreeClassifier

# [2] Create a classification decision tree classifier object as 'dtc':
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)

# [3] Train the decision tree classifier.
dtc = dtc.fit(os_data_X, os_data_y)

# [4] Predict the response for the test data set.
y_pred = dtc.predict(X_test)


# ### 6. Check the accuracy of the model

# In[8]:


# [1] Import scikit-learn metrics module for accuracy calculation:
from sklearn.metrics import confusion_matrix

# [2] Use the 'print()' function to display the confusion matrix results:
print(confusion_matrix(y_test, y_pred))

# [3] Calculate and print:
# [3a] Metrics for 'accuracy'.
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
# [3b] Metrics for 'precision'.
print('Precision:', metrics.precision_score(y_test, y_pred))
# [3c] Metrics for 'recall'.
print('Recall:', metrics.recall_score(y_test, y_pred))


# In[9]:


# Alternate method to generate the classification report.
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### 7. Visualise the decision tree

# In[10]:


# [1] Import matplotlib to create a visualisation
# and the 'tree' package from sklearn:
import matplotlib.pyplot as plt
from sklearn import tree

# [2] Plot the decision tree to create the visualisation:
fig, ax = plt.subplots(figsize = (10, 10))
tree.plot_tree(dtc, fontsize = 10)

# [3] Print the plot with plt.show().
plt.show()


# In[11]:


# Change the levels displayed on the decision tree
# by adjusting the value of max_depth:
dtc = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, random_state = 1)


# ## Regression decision tree

# ### 1. Import and read the data set

# In[14]:


# [1] Import all necessary libraries:
import pandas as pd
import numpy as np
import scipy as scp
import sklearn
# Note: Provides classes and functions to estimate many
#different statistical methods.
import statsmodels.api as sm  

from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split

# Note: Indicates situations that aren’t necessarily exceptions.
import warnings  
# [1a] Filter out any warning messages.
warnings.filterwarnings("ignore")

# [2] Read the CSV file/data set.
df = pd.read_csv('Ecommerce data.csv')

# [3] Print a summary of the DataFrame to sense-check it.
df.info()


# ### 2. Build and fit the model

# In[15]:


# [1] Specify that the column Median_s
# should be moved into a separate DataFrame.
cols = df.columns[df.columns != 'Median_s']

# [2] Specify 'X' as the independent variables
# and 'y' as the dependent variable.
X = df[cols]
y = df['Median_s']

# [3] Split the data into train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 0)

# [4] Import the 'DecisionTreeRegressor' class from sklearn.
from sklearn.tree import DecisionTreeRegressor

# [5] Create the 'DecisionTreeRegressor' class
# (which has many parameters; input only #random_state = 0):
regressor = DecisionTreeRegressor(random_state = 0)

# [6] Fit the regressor object to the data set.
regressor.fit(X_train, y_train)


# ### 3. Check the accuracy of the model

# In[17]:


# [1] Import the necessary packages:
from sklearn import metrics
import math

# [2] Predict the response for the test data set.
y_pred = regressor.predict(X_test)

# [3] Specify to print the MAE and MSE (to evaluate the accuracy of the new model):
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
# [3b] Calculate the RMSE.
print('Root Mean Squared Error: ',
     math.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ## Classification random forest

# ### 1. Import and read the data set

# In[2]:


# [1] Import all the ncessary packages:
import pandas as pd
import numpy as np
import scipy as scp
import sklearn
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings # Note: Indicates situations that aren't necessarily exceptions.
warnings.filterwarnings("ignore") # [1a] Filter out any warning messages.


# In[3]:


# [2] Read the CSV file.
df = pd.read_csv("Customer_data.csv")

# [3] Print a summary of the DataFrame to sense-check it.
df.info()


# ### 2. Update the variables in the 'Edu' column

# In[4]:


# [1] Update all the details of the education column:
df['Edu'][df['Edu'].str.contains('basic')] = 'pre-school'
df['Edu'][df['Edu'].str.contains('university')] = 'uni'
df['Edu'][df['Edu'].str.contains('high')] = 'high-school'
df['Edu'][df['Edu'].str.contains('professional')] = 'masters'
df['Edu'][df['Edu'].str.contains('illiterate')] = 'other'
df['Edu'][df['Edu'].str.contains('unknown')] = 'other'

# [2] Display all the unique values/check changes.
df['Edu'].unique()


# ### 3. Create dummy variables

# In[5]:


# [1] Name new DataFrame and convert categorical variables to dummy variables:
cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# [2] Use the for loop keyword to specify whata actions to apply to all the 'var' items:
# [2a] Specify what nneds to apply to all the variables.
for var in cat_vars:
    cat_list = 'var' + '_' + var
    # [2b] Specify details of the categorical list.
    cat_list = pd.get_dummies(df[var], prefix = var)
    # [2c] Indicate the joining of the DataFrames.
    df1 = df.join(cat_list)
    # [2d] Set the old DataFrame with new df with dummy values.
    df = df1
    
    cat_vars = ['Occupation', 'Status', 'Edu', 'House', 'Loan', 'Comm',
           'Month', 'DOW', 'Last_out']

# [4] Set the temporary DataFrame and add values.
df_vars = df.columns.values.tolist()
# [5] Indicate what columns are kept.
to_keep = [i for i in df_vars if i not in cat_vars]

# [6] Define a new DataFrame.
df_fin = df[to_keep]

# [7] Print the column.
df_fin.columns.values


# ### 4. Balance the data

# In[6]:


# [1] Create a DataFrame to use as df_fin and replace missing valeus with 0.
df_fin = df_fin.fillna(0)

# [2] Select necessary columns:
nec_cols = ['Status_divorced', 'Status_married',
            'Status_single', 'Status_unknown', 
            'Edu_high-school', 'Edu_masters', 
            'Edu_other', 'Edu_pre-school', 
            'Edu_uni', 'House_no', 'House_unknown',
            'House_yes', 'Loan_no', 'Loan_unknown', 
            'Loan_yes', 'DOW_fri', 'DOW_mon']

X = df_fin[nec_cols]
y = df_fin['Target']

# [3] Create a new DataFrame and
# [3a] apply SMOTE as the target variable is not balanced.
os = SMOTE(random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 0)

# [4] Specify column values.
columns = X_train.columns
# [5] Specify the new data sets.
os_data_X, os_data_y = os.fit_resample(X_train, y_train)

# [6] Create two DataFrames for X and y:
os_data_X = pd.DataFrame(data = os_data_X, columns = columns)
os_data_y = pd.DataFrame(data = os_data_y, columns = ['Target'])

# [7] Print/check the DataFrame:
print('Length of oversampled data is ', len(os_data_X))

os_data_y


# In[7]:


# Determine if values in a column is balanced.
os_data_y['Target'].value_counts()


# ### 5. Build and apply the random forest model

# In[8]:


# [1] Import the 'RandomForestClassifier' package.
from sklearn.ensemble import RandomForestClassifier

# [2] Create a forest object based on the 'RandomForestClassifier':
forest = RandomForestClassifier(n_estimators = 200, criterion = 'gini',
                               min_samples_split = 2, min_samples_leaf = 2,
                               max_features = 'auto', bootstrap = True,
                               n_jobs = -1, random_state = 42)

# [3] Train and predict the model:
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

# [4] Import scikit-learn metrics module for accuracy calculation.
from sklearn import metrics

# [5] Model accuracy, how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))


# ### 6. Visualise the random forest model

# In[10]:


# [1] Import the necessary packages.
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

# [2] Plot the visualisation.
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 4), dpi = 800)
tree.plot_tree(forest.estimators_[0], filled = True)

plt.show()


# ### 7. Check the accuracy of the model

# In[12]:


# [1] Import the necessary package:
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# [2] Print a report on the model's accuracy:
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# ### 8. Determine feature significance

# In[14]:


# [1] Import necessary packages:
import time
import numpy as np

# [2] Create a list of feature names:
feature_names = [f'feature {i}' for i in range(X_train.shape[1])]

# [3] Start measuring the time required to construct the random forest model:
start_time = time.time()
# Determine feature importance.
importances = forest.feature_importances_

# [4] Summarise the feature importance:
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)

# [5] Calculate the elapsed time.
elapsed_time = time.time() - start_time
print(f'Elapsed time to compute the importances: {elapsed_time:.3f} seconds')

# [6] Create a data strucure to store the importances:
forest_importances = pd.Series(importances, index = feature_names)

# [7] Construct a bar graph:
fig, ax = plt.subplots(figsize = (12, 12))
forest_importances.plot.bar(yerr = std, ax = ax)
# [7a] Set the title for the graph.
ax.set_title("Feature importances")
# [7b] Set the label for the y-axis.
ax.set_ylabel("Mean decrease in impurity")
# [7c] Adjust padding between and around subplots.
fig.tight_layout()


# ## Regression random forests

# ### 1. Import and read the data set

# In[15]:


# [1] Import all necessary libraries:
import pandas as pd 
import numpy as np 
import scipy as scp
import sklearn
# Note: Provides classes and functions to estimate many different statistical methods.
import statsmodels.api as sm  

from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

# Note: Indicates situations that aren’t necessarily exceptions.
import warnings  
# [1a] Filter out any warning messages.
warnings.filterwarnings("ignore")  

# [2] Read the provided CSV file/data set.
df = pd.read_csv("Ecommerce data.csv")  

# [3] Print a summary of the DataFrame to sense-check it.
df.info()  


# ### 2. Build and fit the model

# In[16]:


# Prepare the data by indicating all the rows and columns for the RRF:
X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values


# In[17]:


# [1] Import the 'train_test_split' package:
from sklearn.model_selection import train_test_split

# [2] Split the data set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 0)

# [3] Import the random forest regressor class:
from sklearn.ensemble import RandomForestRegressor

# [4] Create the regressor object:
regressor = RandomForestRegressor(n_estimators = 5,
                                 random_state = 0,
                                 n_jobs = 2)

# [5] Fit the 'regressor' to the data set.
regressor.fit(X_train, y_train)

# [6] Set 'y_pred'.
y_pred = regressor.predict(X_test)


# ### 3. Check the accuracy of the model

# In[18]:


# Import the metrics package.
from sklearn import metrics

# [2] Calculate and display the metrics:
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# # 

# # 2.3 Clustering with Python

# ## K-means clustering

# ### 1. Prepare your workstation

# In[1]:


# [1] Import libraries.
import pandas as pd
import numpy as np

# [2] Read the data file.
df_fb = pd.read_csv("FB.csv")

# [3] View the DataFrame.
print(df_fb.shape)
print(df_fb.dtypes)
df_fb.head()


# ### 2. Evaluate the variables

# In[2]:


# Display a summary of the numeric variables.
df_fb.describe()


# ### 3. Drop unneeded columns

# In[3]:


# [1] Display the output and
# [2] return the length of the data structure with unique values:
print(len(df_fb['fb_id'].unique()))
print(len(df_fb['medium'].unique()))
print(len(df_fb['published'].unique()))


# In[4]:


# [1] Employ the drop() function and
# [2] indicate the elements to drop:
df_fb.drop(['fb_id', 'published'], axis = 1, inplace = True)

# [3] Display the column names of the DataFrame.
df_fb.columns


# In[5]:


# [1] Import the necessary packages:
from matplotlib import pyplot as plt
import seaborn as sns

# [2] Generate the overall frame and [2a, b] the bars for the plot:
plt.figure(figsize = (12, 12))
ax = sns.countplot(x = 'medium', data = df_fb)
sns.set(font_scale = 1.5)
sns.set_style('white')

# [3] Specify the plot title, x-axis label, and the y-axis label:
plt.title('Distribution of medium')
plt.xlabel('Type of medium')
plt.ylabel('Frequency')

# [4] Annotate the bars with values:
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()),
                (p.get_x() + 0.1, p.get_height() + 50), 
                va = 'center')


# ### 4. Specify the target variable

# In[6]:


# Define the independent variable.
X = df_fb
# Define the dependent variable.
y = df_fb['medium']


# In[7]:


# [1] Import the LbelEncoder class:
from sklearn.preprocessing import LabelEncoder

# [2] Create an object from the class.
le = LabelEncoder()

# [3] Modify the DataFrame column 'medium' with integer values:
X['medium'] = le.fit_transform(X['medium'])

y = le.transform(y)


# In[8]:


# Check that label encoding happened correctly using the 'info()' method:
X.info()


# In[9]:


# Check that label happened correcly using the 'head()' method.
X.head()


# ### 5. Normalise the data set

# In[10]:


# [1] Create an list with the columns labels from X:
x_cols = X.columns

# [2] Import the MinMaxScaler class.
from sklearn.preprocessing import MinMaxScaler

# [3] Create the object from 'MinMaxScaler'.
ms = MinMaxScaler()

# [4] Modify X to scale values between 0 and 1.
X = ms.fit_transform(X)

# [5] Set X as equal to a new DataFrame.
X = pd.DataFrame(X, columns = [x_cols])

# [6] Check the contents of the modifiend DataFrame.
X.head()


# ### 6. Apply the clustering algorithm

# In[11]:


# [1] Import the KMeans class.
from sklearn.cluster import KMeans

# [2] Create the object and [2a] specify the parameters:
kmeans = KMeans(n_clusters = 2, random_state = 0)

# [3] Fir the k-means to the data set.
kmeans.fit(X)


# In[12]:


# Indicate 'kmeans()' applies to 'cluster_centers'.
kmeans.cluster_centers_


# In[13]:


# Check the inertia for the data set.
kmeans.inertia_


# ### 7. Evaluate the output

# In[14]:


# [1] Extract the labels from the k-means.
labels = kmeans.labels_

# [2] Add up correctly labelled instances.
correct_labels = sum(y == labels)

# [3] Display the output.
print('Result: %d out of %d samples were correctly labelled.' % (correct_labels,
                                                                y.size))


# ### 8. Improve the accuracy (elbow method)

# In[15]:


# [1] Create an empty list.
cs = []

# [2] Employ a loop to test cluster sizes:
for i in range(1, 11):
    # [3a] Create object k-means.
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                   max_iter = 300, n_init = 100,
                   random_state = 0)
    # [3b] Apply the fit() method.
    kmeans.fit(X)
    # [3c] Add the inertia value.
    cs.append(kmeans.inertia_)

# [4] Create a plot.
plt.plot(range(1, 11), cs)
# [4a] Speciy the title, x-axis label and y-axis label.
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')

# [5] Display the plot.
plt.show()

# From the elbow method we can take the optimal number as 4.


# In[17]:


# [1] Create a k-means object with three cluster:
kmeans = KMeans(n_clusters = 3, random_state = 0)

# [2] Apply 'fit()', using the DataFrame, to the k-means object.
kmeans.fit(X)

# [3] Check how many of the samples were correctly labelled:
labels = kmeans.labels_
correct_labels = sum(y == labels)

# [4] Display the accuracy score:
print('Result: %d out of %d samples were correctly labelled.' % (correct_labels,
                                                                y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels / float(y.size)))


# In[20]:


# [1] Create a k-means object with four clusters:
kmeans = KMeans(n_clusters = 4, random_state = 0)

# [2] Apply 'fit()', using the DataFrame, to the k-means object.
kmeans.fit(X)

# [3] Check how many of the samples were corectly labelled:
labels = kmeans.labels_
correct_labels = sum(y == labels)

# [4] Display the accuracy score:
print('Result: %d out of %d samples were correctly labelled.' % (correct_labels,
                                                                y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels / float(y.size)))


# ### 9. Visualise the clusters

# In[21]:


# [1] Create the figure area.
fig = plt.figure(figsize = (26, 6))

# [2] Create a 3D projection area.
ax = fig.add_subplot(131, projection = '3d')

# [3] Create a 3D scatter plot and specify the data source for each axis:
ax.scatter(df_fb['reactions'], df_fb['share'], df_fb['like'],
          c = labels, s = 15)

# [4] Set the label for each dimension:
ax.set_xlabel('Reactions')
ax.set_ylabel('Share')
ax.set_zlabel('Like')

# [5] Show the plot.
plt.show()


# In[22]:


# [1] Create the figure area.
fig = plt.figure(figsize = (26, 6))

# [2] Create a 3D projection area.
ax = fig.add_subplot(131, projection = '3d')

# [3] Create a 3D scatter plot and specify the data source for each axis:
ax.scatter(df_fb['comments'], df_fb['share'], df_fb['love'],
          c = labels, s = 15)

# [4] Set the label for each dimension:
ax.set_xlabel('Comments')
ax.set_ylabel('Share')
ax.set_zlabel('Love')

# [5] Show the plot.
plt.show()


# In[ ]:




