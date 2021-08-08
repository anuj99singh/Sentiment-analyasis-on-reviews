#!/usr/bin/env python
# coding: utf-8

# # Reading Data (MileStone - 1)

# In[2]:


import pandas as pd
import streamlit as st


# In[2]:


df= pd.read_csv("amazonLabelled.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


# In[9]:


def find(x):
    ans=0
    for sen in x:
        if (sen=='Positive'):
            ans=ans+1
    return ans 


# In[10]:


Positive = find(df["Sentiment"])
print(Positive)


# In[11]:


Negative = len(df.Sentiment)-Positive
print(Negative)


# In[12]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
Sentiment = ['Negative', 'Positive']
data = [Negative, Positive]
ax.bar(Sentiment, data)
plt.xlabel('Sentiment')
plt.ylabel('Data')
plt.title(" Sentiment Graph")
plt.show()


# # Milestone - 2
# # Removing Punctuation

# In[13]:


punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def remove_puc(text):
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text


# In[14]:


df["punctuation_removed"] = df["Feedback"].apply(lambda x: remove_puc(x))


# In[15]:


df.punctuation_removed[0]


# In[16]:


df.head()


# # Tokenization using NLTK
# 

# In[17]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[18]:


from nltk.tokenize import sent_tokenize

df["nltk_token"] = df["punctuation_removed"].apply(lambda x: word_tokenize(x.lower()))

df.nltk_token[0]


# In[19]:


df.head()


# # Removing Stop_Words

# In[20]:


df["StopWords_Removed"] = df['nltk_token'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])


# In[21]:


df.head()


# # Vectorization

# In[22]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import xgboost, textblob, string
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


# In[23]:


classes = ['Negative', 'Positive']


# In[24]:


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['punctuation_removed'], df['Sentiment'])
# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[25]:


count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(df['punctuation_removed'])
# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)


# # Tf-idf

# In[26]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features=5000)
tfidf_vect.fit(df['punctuation_removed'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)


# In[27]:


# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df['punctuation_removed'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)


# In[28]:


# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df['punctuation_removed'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)


# In[29]:


def train_model(classifier, vector_train, label, vector_valid):
    classifier.fit(vector_train, label)
    predictions = classifier.predict(vector_valid)
    return classification_report(predictions, valid_y ,target_names=classes)


# # Naive Bayes

# In[30]:


# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y,xvalid_count)
print ("NB, Count Vectors: \n", accuracy)
print("------------------------------------------------")


# In[31]:


# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y,xvalid_tfidf)
print ("NB, WordLevel TF-IDF: \n", accuracy)
print("------------------------------------------------")


# In[32]:


# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y,xvalid_tfidf_ngram)
print ("NB, N-Gram Vectors: \n", accuracy)
print("------------------------------------------------")


# In[33]:


# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars,train_y, xvalid_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: \n", accuracy)
print("------------------------------------------------")


# # Linear Classifier

# In[34]:


# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y,xvalid_count)
print( "LR, Count Vectors: \n", accuracy)
print("------------------------------------------------")


# In[35]:


# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y,xvalid_tfidf)
print( "LR, WordLevel TF-IDF: \n", accuracy)
print("------------------------------------------------")


# In[36]:


# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram,train_y, xvalid_tfidf_ngram)
print( "LR, N-Gram Vectors: \n", accuracy)
print("------------------------------------------------")


# In[37]:


# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(),xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print( "LR, CharLevel Vectors: \n", accuracy)
print("------------------------------------------------")


# # Support Vector Machine(SVM)

# In[38]:


# SVM on Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print( "SVM, Count Vectors: \n", accuracy)
print("------------------------------------------------")


# In[39]:


# SVM on Word Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print( "SVM, WordLevel TF-IDF: \n", accuracy)
print("------------------------------------------------")


# In[40]:


# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y,xvalid_tfidf_ngram)
print( "SVM, N-Gram Vectors: \n", accuracy)
print("------------------------------------------------")


# In[41]:


# SVM on Character Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y,xvalid_tfidf_ngram_chars)
print( "SVM, CharLevel Vectors: \n", accuracy)
print("------------------------------------------------")


# # Random Forest

# In[42]:


# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y,xvalid_count)
print( "RF, Count Vectors: \n", accuracy)
print("------------------------------------------------")


# In[43]:


# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y,xvalid_tfidf)
print( "RF, WordLevel TF-IDF: \n", accuracy)
print("------------------------------------------------")


# In[44]:


# RF on Ngram Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram,train_y, xvalid_tfidf_ngram)
print( "RF, N-Gram Vectors: \n", accuracy)
print("------------------------------------------------")


# In[45]:


# RF on Character Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(),xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print( "RF, CharLevel Vectors: \n", accuracy)
print("------------------------------------------------")


# # Extreme Gradient Boosting

# In[46]:


# Extreme Gradient Boosting on Count Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y,xvalid_count.tocsc())
print ("Xgb, Count Vectors: \n", accuracy)
print("------------------------------------------------")


# In[47]:


# Extreme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y,xvalid_tfidf.tocsc())
print ("Xgb, WordLevel TF-IDF: \n", accuracy)
print("------------------------------------------------")


# In[48]:


# Extreme Gradient Boosting on Ngram Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(),train_y, xvalid_tfidf_ngram)
print( "Xgb, N-Gram Vectors: \n", accuracy)
print("------------------------------------------------")


# In[49]:


# Extreme Gradient Boosting on Character Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: \n", accuracy)
print("------------------------------------------------")

