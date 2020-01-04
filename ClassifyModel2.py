#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import os
import sys
import time
import re
import string
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.ensemble import RandomForestClassifier


from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from sklearn.pipeline import Pipeline

from library import TweetCleaner
from library import CustomPipeline



# In[2]:

		


# In[5]:

		
df = pd.read_csv('df_all3.csv')
X_dataset = df[["tweet"]].values
y_dataset = df[["label"]].values



tweet_cleaner = TweetCleaner()
#X_dataset = tweet_cleaner.fit_transform(X_dataset)
# In[9]:

count_vectorizer = CountVectorizer()
#X_dataset = count_vectorizer.fit_transform(X_dataset)



tfidf_transformer = TfidfTransformer()
#X_dataset = tfidf_transformer.fit_transform(X_dataset)

#print(X_dataset)
#print(y_dataset)





y_val_total = []
NB_prediction_total = []

kfold = KFold(n_splits=10)

	
		
# for train, val in kfold.split(X_dataset):
# 	X_train = X_dataset[train]
# 	y_train = y_dataset[train]

# 	X_val = X_dataset[val]
# 	y_val = y_dataset[val]
	
	
# 	model_1 = CustomPipeline(MultinomialNB())

# 	model_1.fit(X_train,y_train)

# 	NB_prediction = model_1.predict(X_val)
	
# 	print(NB_prediction)
# 	print(y_val)
# 	NB_prediction_total = NB_prediction_total + NB_prediction.tolist()
# 	y_val_total = y_val_total + y_val.T[0].tolist()


# print(NB_prediction_total)
# print(y_val_total)
# print('gridNB')
# print(classification_report(y_val_total,NB_prediction_total))
# print(accuracy_score(y_val_total,NB_prediction_total))

# model_final = CustomPipeline(MultinomialNB())
# model_final.fit(X_dataset,y_dataset)
# pickle.dump(model_final, open('nb.sav', 'wb'))



# In[ ]:

# df2 = pd.read_csv('df_total.csv')
# X1_dataset = df[['tweet']].values
# y1_dataset = df[['label']].values

# y1_val_total = []
# NB1_prediction_total = []

# for train,val in kfold.split(X1_dataset) :
# 	X1_train = X1_dataset[train]
# 	y1_train = y1_dataset[train]

# 	X1_val = X1_dataset[val]
# 	y1_val = y1_dataset[val]
	
	
# 	model_2 = CustomPipeline(MultinomialNB())

# 	model_2.fit(X1_train,y1_train)

# 	NB1_prediction = model_2.predict(X1_val)
	
# 	print(NB1_prediction)
# 	print(y1_val)
# 	NB1_prediction_total = NB1_prediction_total + NB1_prediction.tolist()
# 	y1_val_total = y1_val_total + y1_val.T[0].tolist()

# print(NB1_prediction_total)
# print(y1_val_total)
# print('gridNB untuk model pertama')
# print(classification_report(y1_val_total,NB1_prediction_total))
# print(accuracy_score(y1_val_total,NB1_prediction_total))

model_final = CustomPipeline(SVC(C=10, gamma=1, kernel='linear'))
model_final.fit(X_dataset,y_dataset)
pickle.dump(model_final, open('svm2.sav', 'wb'))