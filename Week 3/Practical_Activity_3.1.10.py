#!/usr/bin/env python
# coding: utf-8

# ## Practical Activity 3.1.10

# ### 1. Prepare your workstation

# In[1]:


# Install vaderSentiment tool.
get_ipython().system('pip install VaderSentiment')


# In[2]:


# Import the necessary class.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create an object from the class.
sia = SentimentIntensityAnalyzer()


# ### 2. Define the input data

# In[5]:


# Define the input data.
sentences = [
"Excellent customer service. Also loved the amazing showroom where you can \
get a real feel of the top quality furniture and get inspiration for room \
designs.",
"I found the sales person to be quite rude and stuck up... don't think she \
realised I know all the suppliers she uses. I got the sofa & chair elsewhere. \
I will not be coming back",
"Thank you for finding replacement crystals for my chandelier.",
"I have worked with the team in several houses. All work was carried out on \
time and to budget."]


# In[7]:


# Display the data.
print(sentences)


# ### 3. Apply sentiment analysis

# In[9]:


# Apply sentiment analysis.
for sentence in sentences:
    score = sia.polarity_scores(sentence)["compound"]
    print(f'The sentiment value of the sentence: "{sentence}" is: {score}')


# ### 4. Calculate the output

# In[10]:


# Calculate the percentage of the output.
for sentence in sentences:
    print(f'For the sentence "{sentence}"')
    # Calculate the scores.
    polarity = sia.polarity_scores(sentence)
    pos = polarity['pos']
    neu = polarity['neu']
    neg = polarity['neg']
    
# Display the results.
print(f'The percentage of positive sentiment in: "{sentence}" is : {round(pos * 100, 2)} %')
print(f'The percentage of neutral sentiment in: "{sentence}" is : {round(neu * 100, 2)} %')
print(f'The percentage of negative sentiment in: "{sentence}" is : {round(neg * 100, 2)} %')
print("=" * 50)


# In[ ]:




