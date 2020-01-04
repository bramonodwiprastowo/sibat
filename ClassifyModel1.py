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

		
df = pd.read_csv('df_total.csv')
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





# y_val_total = []
# MNB_prediction_total = []
# CNB_prediction_total = []
# BNB_prediction_total = []

# kfold = KFold(n_splits=10)

	
		
# for train, val in kfold.split(X_dataset):
#     X_train = X_dataset[train]
#     y_train = y_dataset[train]
    
#     X_val = X_dataset[val]
#     y_val = y_dataset[val]
	
#     #param_grid = {'alpha': [0.01,0.1,1,10,100]}
#     model1 = CustomPipeline(MultinomialNB(alpha=0.01))
#     model1.fit(X_train,y_train)
#     MNB_prediction = model1.predict(X_val)
#     # model2 = CustomPipeline(GridSearchCV(ComplementNB(),param_grid,refit=True,verbose=1))
#     # model2.fit(X_train,y_train)
#     # CNB_prediction = model2.predict(X_val)
    
#     # model3 = CustomPipeline(GridSearchCV(BernoulliNB(),param_grid,refit=True,verbose=1))
#     # model3.fit(X_train,y_train)
#     # BNB_prediction = model3.predict(X_val)
    
#     print(MNB_prediction)
#     print(y_val)
#     MNB_prediction_total = MNB_prediction_total + MNB_prediction.tolist()
#     y_val_total = y_val_total + y_val.T[0].tolist()

#     # print(CNB_prediction)
# 	# print(y_val)
# 	# CNB_prediction_total = CNB_prediction_total + CNB_prediction.tolist()
# 	# y_val_total = y_val_total + y_val.T[0].tolist()

#     # print(BNB_prediction)
# 	# print(y_val)
# 	# BNB_prediction_total = BNB_prediction_total + BNB_prediction.tolist()
# 	# y_val_total = y_val_total + y_val.T[0].tolist()


# print(MNB_prediction_total)
# print(y_val_total)
# print('Multinomial NB')
# #print(model1.best_params_)
# print(classification_report(y_val_total,MNB_prediction_total))
# print(accuracy_score(y_val_total,MNB_prediction_total))

# print(CNB_prediction_total)
# print(y_val_total)
# print('Complement NB')
# print(model2.best_params_)
# print(classification_report(y_val_total,CNB_prediction_total))
# print(accuracy_score(y_val_total,CNB_prediction_total))

# print(BNB_prediction_total)
# print(y_val_total)
# print('Bernoulli NB')
# print(model3.best_params_)
# print(classification_report(y_val_total,BNB_prediction_total))
# print(accuracy_score(y_val_total,BNB_prediction_total))

# model_final = CustomPipeline(BernoulliNB(alpha=10))
# model_final.fit(X_dataset,y_dataset)
# pickle.dump(model_final, open('nb1.sav', 'wb'))

model_final2 = CustomPipeline(SVC(C=100, kernel='rbf', gamma=0.01))
model_final2.fit(X_dataset,y_dataset)
pickle.dump(model_final2, open('svm1.sav', 'wb'))

# model_final3 = CustomPipeline(DecisionTreeClassifier(criterion='entropy', splitter='best'))
# model_final3.fit(X_dataset,y_dataset)
# pickle.dump(model_final3, open('dt1.sav', 'wb'))