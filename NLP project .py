#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install spacy


# In[7]:


import spacy
from heapq import nlargest  # This module provides an implementation of the heap queue algorithm, also known as the priority queue algorithm.


# In[12]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[14]:


text= """
There are broadly two types of extractive summarization tasks depending on what the summarizat
ion program focuses on. The first is generic summarization, which focuses on obtaining a gener
ic summary or abstract of the collection (whether documents, or sets of images, or videos, new
s stories etc.). The second is query relevant summarization, sometimes called query-based summ
arization, which summarizes objects specific to a query. Summarization systems are able to cre
ate both query relevant text summaries and generic machine-generated summaries depending on wh
at the user needs.
An example of a summarization problem is document summarization, which attempts to automatical
ly produce an abstract from a given document. Sometimes one might be interested in generating
a summary from a single source document, while others can use multiple source documents (for e
xample, a cluster of articles on the same topic). This problem is called multi-document summar
ization. A related application is summarizing news articles. Imagine a system, which automatic
ally pulls together news articles on a given topic (from the web), and concisely represents th
e latest news as a summary.
Image collection summarization is another application example of automatic summarization. It c
onsists in selecting a representative set of images from a larger set of images.[13] A summary
in this context is useful to show the most representative images of results in an image collec
tion exploration system. Video summarization is a related domain, where the system automatical
ly creates a trailer of a long video. This also has applications in consumer or personal videos
, where one might want to skip the boring or repetitive actions. Similarly, in surveillance v
ideos, one would want to extract important and suspicious activity, while ignoring all the bor
ing and redundant frames captured.


"""


# In[15]:


#Load english large model from SpaCy
nlp = spacy.load('en_core_web_sm')


# In[1]:


# Tokenization
doc = nlp(text)


# In[22]:


#We will get text words frequencies
word_frequencies = {}
for token in doc:
    # Remove stopwords and punctuations, and also '\n'
    if token.is_stop or token.is_punct or str(token) == '\n':
            continue
    
    # At the first of each word, the word is not existed in the dict
    if token.text not in word_frequencies.keys():
        word_frequencies[token.text] = 1
        
    else:
        word_frequencies[token.text] += 1

 


# In[23]:


# That's words frequencies
print(word_frequencies)


# In[24]:


# Get the count of most frequency item
max_frequency = max(word_frequencies.values())
max_frequency


# In[25]:


#We will normalize these frequencies with max frequency item
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_frequency 

# After normalizing 
print(word_frequencies)


# In[29]:


#Sentences Tokenization
# Store sentences in a list
sentence_tokens = [sent for sent in doc.sents]
for sent in sentence_tokens:
    print(sent)


# In[30]:


#We will get sentences frequencies
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        # Chech if each word is existed in words list
        if word.text.lower() in word_frequencies.keys():
            
            # At the first of each sentence, the sentence is not existed in the dict
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]


# In[31]:


for k, v in sentence_scores.items():
    print(f'Sentence: {k} || Frequence: {v}')


# In[32]:


#We want to summarize the full text to a factor (factor)
factor = 0.3
select_length = int(len(sentence_tokens) * factor)
select_length


# In[34]:


# nlargest function : returns the specified number of largest elements
lg_sents = nlargest(select_length, sentence_scores, key=sentence_scores.get)
lg_sents


# In[35]:


# lg_sents is list contains 'Span' items
for sent in lg_sents:
    print(type(sent))


# In[36]:


# Convert these items to a strings
summary_sents = [word.text for word in lg_sents]
summary_sents


# In[37]:


#The final summary
summary = ' '.join(summary_sents)
print(summary)


# In[ ]:




