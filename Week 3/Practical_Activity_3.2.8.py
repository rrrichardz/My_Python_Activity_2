#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 3.2.8

# ### Prepare your workstation

# In[2]:


# Copy the YAML file and your Twitter keys over to this Jupyter Notebook before you start to work
get_ipython().system('pip install pyyaml')
import yaml
from yaml.loader import SafeLoader
get_ipython().system('pip install twitter')
from twitter import *

# Import the yaml file - remember to specify the whole path and use / between directories
twitter_creds = yaml.safe_load(open('Twitter.yaml', 'r').read())

# Pass your twitter credentials
twitter_api = Twitter(auth = OAuth(twitter_creds['access_token'],
                                  twitter_creds['access_token_secret'],
                                  twitter_creds['api_key'],
                                  twitter_creds['api_secret_key']))

# See if you are connected
print(twitter_api)


# In[3]:


# Run a test with #python
python_tweets = twitter_api.search.tweets(q="#python")

# View output
print(python_tweets)


# ### 1. Test connection

# In[8]:


# Query the term cheesecake
q = {'q':'cheesecake', 'count':100, 'result_type':'recent'}
results = []

while len(results) < 30:
    query = twitter_api.search.tweets(**q)
    q['max_id'] = query['search_metadata']['next_results'].split('&')[0].split('?max_id=')[1]
    results.append(query)
    
# Determine the number of results
len(results)    


# ### 2. Create DataFrames

# In[10]:


# Import Pandas to join the DataFrames.
import pandas as pd

# Concat DataFrames.
results_list_pd = pd.concat([pd.DataFrame(_['statuses'])for _ in results])

# View the output.
results_list_pd.shape


# In[14]:


# Determine values of output.
results_list_values = results_list_pd['text'].values

results_list_values


# ### 3. Investigate tweets

# In[19]:


# Import necessary libraries.
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('stopwords')
nltk.download('words')
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


# In[16]:


# Look at one raw tweet.
results_list_values[1]


# In[17]:


# Split up each tweet into individual words.
results_list_values_token = [word_tokenize(_) for _ in results_list_values]


# In[20]:


# Extract a list of all english words.
all_english_words = set(words.words())


# In[21]:


# Pre-process tweets:
# Get every word.
# Convert it to lowercase.
# Only include if the word is alphanumeric and if it is in the list 
# of English words.
results_list_values_token_nostop = [[y.lower() for y in x if y.lower() not in stop_words and y.isalpha() and y.lower() in all_english_words] for x in results_list_values_token]


# In[22]:


# View output.
results_list_values_token_nostop[1]


# ## Continue to work from below for sentiment analysis

# ## NLTK sentiment analysis 

# ### 1. Import NLTK

# In[23]:


# Import the prebuilt rules and values of the 'vader lexicon'.
nltk.download('vader_lexicon')


# In[24]:


# Import the vader class and create an object of the analyser called darth_vader.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create a variable to store the SIA.
darth_vader = SentimentIntensityAnalyzer()


# In[25]:


# Run through a dictionary comprehension to take every cleaned tweet.
# Next run the polarity score function on the string.
# This will return four values in a dictionary.
results_list_values_token_nostop_polarity = {" ".join(_) : darth_vader.polarity_scores(" ".join(_)) for _ in results_list_values_token_nostop}


# ### 2. Create a Pandas DataFrame

# In[26]:


# Convert the list of dictionary results to a pandas dataframe.
polarity_pd = pd.DataFrame(results_list_values_token_nostop_polarity).T


# In[27]:


# Get the top 5 most positive cleaned tweets related to cheesecake
polarity_pd.sort_values('pos', ascending = 0).head(5)


# In[28]:


# Get the top 5 most negative words related to cheesecake
polarity_pd.sort_values('neg', ascending = 0).head(5)


# In[29]:


# Show the distribution of sentiment values.
polarity_pd['compound'].describe()


# ### 3. Plot the output

# In[30]:


# Import matplotlib for visualisation.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[33]:


# Create barplot visualisation.
_plot = polarity_pd.reset_index()['compound'].sort_values()
_plot.plot(kind = 'bar')
ax1 = plt.axes()
x_axis = ax1.axes.get_xaxis()
x_axis.set_visible(False)

plt.show()
plt.close()


# In[34]:


# Visualise outliers with a boxplot.
_plot = polarity_pd.reset_index()['compound'].sort_values()
_plot.plot(kind = 'box')


# In[ ]:




