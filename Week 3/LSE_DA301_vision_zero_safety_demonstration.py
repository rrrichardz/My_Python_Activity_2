#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# # Vision Zero road safety survey analysis
# 
# This notebook looks at responses to a survey about road safety. The ultimate objective of the data anaysis is to improve driver, pedestrian, and bicyclist transportation safety. What can we extract from the comments left by the respondents that can help us better understand the current sentiment towards road safety? Could we perhaps find any strong indication where improvements need to be made?

# In[5]:


# Import libraries.
import numpy as np
import pandas as pd

import warnings # Note: Indicates situations that aren't necessarily exceptions.
warnings.filterwarnings('ignore') # Filter out any warning messages.
# Import data.
survey_data = pd.read_csv("Vision_Zero_Safety.csv")

# View the data.
print(survey_data.shape)
survey_data.head()


# ## 1. Pre-process the data

# ### Drop rows from the table that do not have any value for their COMMENTS field

# In[6]:


survey_data.dropna(subset = ['COMMENTS'], inplace = True)
survey_data.shape


# ### Change all the words in the comments to lower case

# In[7]:


# Transform data to lowercase.
survey_data['COMMENTS'] = survey_data['COMMENTS'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Preview the results.
survey_data['COMMENTS'].head()


# ### Remove punctuation

# In[8]:


survey_data['COMMENTS'] = survey_data['COMMENTS'].str.replace('[^\w\s]','')
survey_data['COMMENTS'].head()


# ### Drop duplicates from the COMMENTS column

# In[9]:


# Check the number of duplicate values in the COMMENTS column.
survey_data.COMMENTS.duplicated().sum()


# In[10]:


# Drop duplicates.
survey = survey_data.drop_duplicates(subset = ['COMMENTS'])


# In[11]:


# Preview data.
survey.reset_index(inplace = True)
survey.head()


# In[12]:


survey.shape


# As we can see that the rows with empty fields and the duplicate entries in the COMMENTS column have been removed from the dataframe.

# ## 2. Visualise the most frequently used words

# In[13]:


# String all the comments together in a single variable.
all_comments = ''
for i in range(survey.shape[0]):
    # Add each comment.
    all_comments = all_comments + survey['COMMENTS'][i]


# In[14]:


# Import along with matplotlib and seaborn for visualisation.
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[15]:


# Set the colour palette.
sns.set(color_codes = True)

# Create a WordCloud object.
wordcloud = WordCloud(width = 1600, height = 900, 
                background_color = 'white',
                colormap = 'plasma', 
                stopwords = 'none',
                min_font_size = 10).generate(all_comments) 
  
# Plot the WordCloud image.                       
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# ## 3. Find the most frequently used words with tokenisation

# In[16]:


# Import nltk and download nltk's resources to assist with tokenisation.
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


# ### Tokenisation

# In[17]:


# Tokenise the words.
survey['tokens'] = survey['COMMENTS'].apply(word_tokenize)
# Preview the results.
survey['tokens'].head()


# In[18]:


# Combine all tokens into one list.
all_tokens = []
for i in range(survey.shape[0]):
    all_tokens = all_tokens + survey['tokens'][i]


# In[19]:


# Compute the tokens with the maximum frequency.
from nltk.probability import FreqDist
# Calculate the frequency distribution.
fdist = FreqDist(all_tokens)
fdist


# In[20]:


# Filter out tokens that are neither alphabets or numbers.
# (to eliminate punctuation marks etc)
tokens1 = [word for word in all_tokens if word.isalnum()]


# ### Eliminate stopwords

# In[21]:


# Download the nltk resource, import the method, and extract stopwords in the English language.
nltk.download ('stopwords')
from nltk.corpus import stopwords

# Create a set of Enligsh stop words.
english_stopwords = set(stopwords.words('english'))


# In[22]:


# Create a list of tokens without stop words.
tokens2 = [x for x in tokens1 if x.lower() not in english_stopwords]


# In[23]:


# Define an empty string variable.
tokens2_string = ''
for value in tokens2:
    # Add each filtered token word to the string.
    tokens2_string = tokens2_string + value + ' '


# ### Visualise the tokens of relevance in a word cloud

# In[24]:


# Create a word cloud object.
wordcloud = WordCloud(width = 1600, height = 900, 
                background_color = 'white', 
                colormap = 'plasma', 
                min_font_size = 10).generate(tokens2_string) 

# Plot the WordCloud image                        
plt.figure(figsize = (16, 9), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()


# ## 4. Count the top 15 words that appear most often

# In[25]:


# View the frequency distribution.
fdist1 = FreqDist(tokens2)
fdist1


# In[26]:


'''top 15 commonly used words across the comments.'''
# Import Counter.
from collections import Counter

# Generate DF out of Counter.
counts = pd.DataFrame(Counter(tokens2).most_common(15),
                      columns = ['Word', 'Frequency']).set_index('Word')
# Display the result.
counts


# In[29]:


# Generate bar graph.
# Set the plot type.
ax = counts.plot(kind = 'barh', figsize = (16, 9), fontsize = 12,
                 colormap = 'plasma')
# Set the labels.
ax.set_xlabel("Count", fontsize = 12)
ax.set_ylabel("Word", fontsize = 12)
ax.set_title("Safety survey responses: Count of the 15 most frequent words",
             fontsize = 20)

# Add annotations.
for i in ax.patches:
    ax.text(i.get_width() + .41, i.get_y() + .1, str(round((i.get_width()), 2)),
            fontsize = 12, color = 'red')


# In[ ]:




