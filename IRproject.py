#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import arff

arff_file_path = '/Users/sabathaher/Downloads/IRSdataset.arff'
with open(arff_file_path, 'r', encoding='latin-1') as f:
    arff_data = arff.load(f)

# Extract data from the dictionary
data = arff_data['data']
meta = arff_data['attributes']

# Convert data to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())


# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[3]:


def remove_punctuation(text):
    # Remove punctuation using regex
    return re.sub(r'[^\w\s]', '', text)

def remove_digits(text):
    # Remove digits using regex
    return re.sub(r'\d', '', text)

def remove_stopwords(text):
    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


# In[4]:


import re


# In[5]:


# Assuming 'text' column is at position 0
df.iloc[:, 0] = df.iloc[:, 0].apply(remove_punctuation)
df.iloc[:, 0] = df.iloc[:, 0].apply(remove_digits)
df.iloc[:, 0] = df.iloc[:, 0].apply(remove_stopwords)


# In[6]:


df


# In[7]:


from nltk.stem import PorterStemmer


# In[8]:


def apply_stemming(text):
    # Apply stemming using NLTK's Porter Stemmer
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)


# In[9]:


df.iloc[:, 0] = df.iloc[:, 0].apply(apply_stemming)


# In[10]:


df


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


corpus = df.iloc[:, 0].tolist()
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df.drop(columns=[0]), bow_df], axis=1)


# In[13]:


df


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df = pd.concat([ tfidf_df], axis=1)
df.to_csv('tfidf.csv',index=0)


# In[16]:


df


# In[17]:


import numpy as np


# In[18]:


from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


# In[19]:


tokenized_corpus = [word_tokenize(text) for text in corpus]
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)


# In[20]:


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector

def get_avg_feature_vectors(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(words, model, vocabulary, num_features) for words in corpus]
    return np.array(features)


# In[21]:


# Get the average Word2Vec feature vectors
word2vec_features = get_avg_feature_vectors(tokenized_corpus, word2vec_model, num_features=100)

# Convert the Word2Vec features to a DataFrame
word2vec_df = pd.DataFrame(word2vec_features, columns=[f'word2vec_{i}' for i in range(100)])

# Concatenate the Word2Vec DataFrame with the original DataFrame
df = pd.concat([df, word2vec_df], axis=1)


# In[22]:


df


# In[ ]:




