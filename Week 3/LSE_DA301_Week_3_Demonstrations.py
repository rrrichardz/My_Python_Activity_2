#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# ## Week 3: Sentiment analysis with natural language processing

# This week, we will learn how to analyse textual data to determine the sentiment of customers or users towards a service or brand, inform business decision-making, and take action to improve customer service. We will learn how text analytics (which includes sentiment analysis and is an AI technology that uses natural language processing (NLP)) to extract information from unstructured, textual data. Such data sources include online customer reviews, commentary on social media platforms, and feedback in customer surveys. 
# 
# This is your notebook. Use it to follow along with the demonstrations, test ideas and explore what is possible. The hands-on experience of writing your own code will accelarate your learning!
# 
# For more tips: https://jupyter-notebook.readthedocs.io/en/latest/ui_components.html

# # 3.1 Sentiment analysis with natural language processing

# ## Pre-processing: Tokenisation

# ### 1. Prepare your workstation

# In[1]:


# Install NLTK library.
get_ipython().system('pip install nltk')

# Import all necessary libraries.
import nltk

# Install the required tokenisation model.
nltk.download('punkt')

# Install the required tokenisation function.
from nltk.tokenize import sent_tokenize


# ### 2. Tokenise the sentences

# In[4]:


# [1] Assign the raw text data to a variable.
text = """We took this ball to the beach and after close to 2 hours to pump it up, we pushed it around for about 10 fun filled minutes. That was when the wind picked it up and sent it huddling down the beach at about 40 knots. It destroyed everything in its path. Children screamed in terror at the giant inflatable monster that crushed their sand castles. Grown men were knocked down trying to save their families. The faster we chased it, the faster it rolled. It was like it was mocking us. Eventually, we had to stop running after it because its path of injury and destruction was going to cost us a fortune in legal fees. Rumor has it that it can still be seen stalking innocent families on the Florida panhandle. We lost it in South Carolina, so there is something to be said about its durability."""

# [2] Tokenise the text data.
tokenised_sentence = sent_tokenize(text)

# [3] Check the result.
print(tokenised_sentence)


# In[5]:


# Import the function.
from nltk.tokenize import word_tokenize

# Tokenise the text data.
tokenised_word = word_tokenize(text)

# Check the result.
print(tokenised_word)


# ### 3. Analyse individual words

# In[6]:


# [1] Import the class.
from nltk.probability import FreqDist

# [2] Create a frequency distribution object.
freq_dist_of_words = FreqDist(tokenised_word)

# [3] Show the five most common elements in the data set.
freq_dist_of_words.most_common(5)


# In[9]:


# [1] Import the package.
import matplotlib.pyplot as plt

# [2] Define the figure and axes.
fig, ax = plt.subplots(dpi = 100)
fig.set_size_inches(12, 12)

# [3] Plot the data.
freq_dist_of_words.plot(30, cumulative = False)

# [4] Set the labels for the axes
ax.set_xlabel('Counts', fontsize = 20)
ax.set_ylabel('Samples', fontsize = 20)

# [5] Display the result.
plt.show()


# ## Pre-processing: Normalisation

# ### 1. Remove stop words

# In[10]:


# [1] Import all necessary libraries.
import nltk

# [2] Download the stopwords.
nltk.download('stopwords')

# [3] Import the package.
from nltk.corpus import stopwords

# [4] Create a set of English stop words.
stop_words = set(stopwords.words('english'))

# [5] Display the set.
print(stop_words)


# In[12]:


# [1] Create an empty list for the filtered words.
filtered_text = []

# [2] Create a tokenised word list.
tokenised_word = word_tokenize(text)

# [3] Filter the tokenised words.
for each_word in tokenised_word:
    if each_word not in stop_words:
        filtered_text.append(each_word)
        
# [4] Display the filtered list.
print('Tokenised list without stop words: {}'.format(filtered_text))


# ### 2. Reduce words using stemming and lemmatisation

# #### Using a Snowball Stemmer algorithm

# In[13]:


# [1] Import the necessary class.
from nltk.stem.snowball import SnowballStemmer

# [2] Download the resource.
nltk.download('wordnet')

# [3] Create a stemming object.
snow_stem = SnowballStemmer(language = 'english')

# [4] Create a list of test words:
words = ['easily', 'durability', 'longest', 'wishing', 'worthwhile', 
         'fantasizing', 'off-putting']

# [5] Apply the stemming process to each word.
for word in words:
    print(word + '--->' + snow_stem.stem(word))


# #### Using a lemmatiser

# In[19]:


# [1] Import the necessary class.
import nltk
from nltk.stem.wordnet import WordNetLemmatizer

# [2] Download the lemma corpus.
nltk.download('omw-1.4')

# [3] Create an instance of the class.
lemmatiser = WordNetLemmatizer()

# [4] Define a text string to test.
text = "I love it when he purchases items that the kids and grandkids can't wait to try out. it's a lot of fun but he easily and accidentally bowls the toddlers over with them, so be careful."

# [5] Create an empty output list.
lemmatised_words_list = []

# Tokenise the string.
tokenised_word = word_tokenize(text)

# [6] Apply lemmatisation to each tokenised word.
for each_word in tokenised_word:
    lem_word = lemmatiser.lemmatize(each_word)
    lemmatised_words_list.append(lem_word)
    
# Display the output list.
print('Lemmatised words list: {}'.format(lemmatised_words_list))


# ## Pre-processing: Noise removal

# In[20]:


# [1] Import the necessary module.
import re

# [2] Define some text.
text = "Perfect! Buying a second. Using it to make a hot air balloon for an escape room adventure. Event in Oct. will share photos."

# [3] Filter out the specified punctuation.
no_punct = re.sub(r'[\.\?\!\,\:\;\"]', '', text)

# [4] Display the filtered text.
print(no_punct)


# In[ ]:





# # 

# # 3.2 Working with textual data

# ## Naive Bayes sentiment classifier model

# ### 1. Prepare your workstation

# In[2]:


# [1] Import the necessary library.
import nltk

# [2] Download the existing movie reviews.
nltk.download('movie_reviews')


# ### 2. Construct a list of documents

# In[5]:


# [1] Import the necessary libraries
from nltk.corpus import movie_reviews
import random

# [2] Construct a nested list of documents.
documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]

# [3] Reorganise the list randomly.
random.shuffle(documents)


# In[6]:


# [1] Create a list of files with negative reviews.
negative_ids = movie_reviews.fileids('neg')

# [2] Create a list of files with positive reviews.
positive_ids = movie_reviews.fileids('pos')

# [3] Display the list lengths.
print(len(negative_ids), len(positive_ids))


# In[7]:


# View the output.
print(movie_reviews.raw(fileids = positive_ids[0]))


# ### 3. Define a feature extractor function

# In[8]:


# [1] Create an object to contain the frequency distribution.
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# [2] Create a list that contains the first 2000 words.
word_features = list(all_words)[:2000]

# [3] Define a function to check if each word is in the set of features.
def document_features(document):
    # Create a set of document words.
    document_words = set(document)
    # Create an empty dictionary of features.
    features = {}
    # Populate the dictionary.
    for word in word_features:
        # Specify whether each feature exists in the set of document words.
        features['contains({})'.format(word)] = (word in document_words)
    # Return the completed dictionary.
    return features


# In[9]:


# Generate a dictionary for the first review.
test_result = document_features(documents[0][0])

for key in test_result:
    print(key, ' : ', test_result[key])


# ### 4. Train the classifier

# In[10]:


# [1] Create a list of feature sets based on the documents list.
featuresets = [(document_features(d), c) for (d, c) in documents]

# [2] Assign items to the train and test sets.
train_set, test_set = featuresets[100:], featuresets[:100]

# [3] Create a classifier object trained on items from the train set.
classifier = nltk.NaiveBayesClassifier.train(train_set)

# [4] Display the accuracy score in comparison with the test set.
print(nltk.classify.accuracy(classifier, test_set))


# ### 5. Interpret the results

# In[11]:


classifier.show_most_informative_features(5)


# In[ ]:





# # 

# # 3.3 Obtaining actionable insights from survey analysis

# In[2]:


"""Find demonstrations on sub practice set"""


# In[ ]:




