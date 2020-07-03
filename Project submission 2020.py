#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis-Twitter data 

# In[ ]:


import requests
import bs4
from bs4 import BeautifulSoup
from requests_oauthlib import OAuth1


# # extracting twitter data

# In[6]:


auth_params = {
    'app_key':'API Key',
    'app_secret':'API secret',
    'oauth_token':'Access token',
    'oauth_token_secret':'Access token secret'
}

# Creating an OAuth Client connection
auth = OAuth1 (
    auth_params['app_key'],
    auth_params['app_secret'],
    auth_params['oauth_token'],
    auth_params['oauth_token_secret']
)


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import Word
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[8]:


data = pd.read_csv('Downloads/text_sentiment.csv')

data = data.drop('author', axis=1)


# In[9]:


data.head()


# In[10]:


data.tail()


# In[11]:


data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)


# In[12]:


data.head()


# In[13]:


data.tail()


# In[14]:


data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))


data['tweet'] = data['tweet'].str.replace('[^\w\s]',' ')

stop = stopwords.words('english')
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


data['tweet'] = data['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['tweet'] = data['tweet'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))


freq = pd.Series(' '.join(data['tweet']).split()).value_counts()[-10000:]


freq = list(freq.index)
data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)



# In[15]:


import nltk
nltk.download('wordnet')


# In[16]:


X_train, X_val, y_train, y_val = train_test_split(data.tweet.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)


# # Model prediction and accuracy

# In[17]:


Tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_Tfidf = Tfidf.fit_transform(X_train)
X_val_Tfidf = Tfidf.fit_transform(X_val)

count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['tweet'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

nb = MultinomialNB()
nb.fit(X_train_Tfidf, y_train)
y_pred = nb.predict(X_val_Tfidf)
print('naive bayes tfidf accuracy %s' % accuracy_score(y_pred, y_val))


lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_Tfidf, y_train)
y_pred = lsvm.predict(X_val_Tfidf)
print('svm using tfidf accuracy %s' % accuracy_score(y_pred, y_val))

logreg = LogisticRegression(C=1)
logreg.fit(X_train_Tfidf, y_train)
y_pred = logreg.predict(X_val_Tfidf)
print('log reg tfidf accuracy %s' % accuracy_score(y_pred, y_val))

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_Tfidf, y_train)
y_pred = rf.predict(X_val_Tfidf)
print('random forest tfidf accuracy %s' % accuracy_score(y_pred, y_val))


# In[18]:


nb = MultinomialNB()
nb.fit(X_train_count, y_train)
y_pred = nb.predict(X_val_count)
print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))

lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))

logreg = LogisticRegression(C=1)
logreg.fit(X_train_count, y_train)
y_pred = logreg.predict(X_val_count)
print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_val))

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_count, y_train)
y_pred = rf.predict(X_val_count)
print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_val))


# # positive and negative tweet analysis

# In[19]:


tweets = pd.DataFrame(['I am very happy shopping with amazon.Ind',
'Offers are great in snapdeal compared with others',
'Success is right around the corner when yoy receive fast delivery',
'Everything is more beautiful when you shop with snapdeal',
'Now this is my worst prodect i have ever received',
'I am tired of receiving used prodects from amazon and they are defective',
'This is quite depressing when u dont get a cash back in amazon',
'a defective product that is received broke my heart. It was a sad day', 'Boring offers at flipkart!','Happy shopping Amazon!'])


tweets[0] = tweets[0].str.replace('[^\w\s]',' ')
from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets[0] = tweets[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
from textblob import Word
tweets[0] = tweets[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


tweet_count = count_vect.transform(tweets[0])


tweet_pred = lsvm.predict(tweet_count)
print(tweet_pred)


# # word cloud

# In[20]:


from wordcloud import WordCloud, STOPWORDS 

data = pd.read_csv('Downloads/text_sentiment.csv')
  
comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in data.sentiment: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:




