import pandas as pd
import numpy as np
import os
import json
import sys
import time
import re
import string
import random
from sklearn.pipeline import Pipeline
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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import datetime
import pickle
#import mysql.connector
#from mysql.connector import Error
#from mysql.connector import errorcode

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener


class TweetCleaner() :
    def transform(self,X) :
        tweet_final = []
        for tweet in X :
            tweet = tweet[0]
            tweet = tweet.lower()
            regex = re.compile('\shttp.+\s')
            tweet = regex.sub('',tweet)
            regex = re.compile('\shttps.+\s')
            tweet = regex.sub('',tweet)
            regex = re.compile('\spic.+\s')
            tweet = regex.sub('',tweet)
            regex = re.compile('\sftp.+\s')
            tweet = regex.sub('',tweet)
            regex = re.compile('[^a-zA-Z0-9]')
            tweet = regex.sub(' ',tweet)
            regex = re.compile('[0-9]+')
            tweet = regex.sub('',tweet)
            regex = re.compile(r'\W*\b\w{1,3}\b')
            tweet = regex.sub('',tweet)
            regex = re.compile('rt\s')
            tweet = regex.sub(' ',tweet)

            replacement_words_list = [line.rstrip('\n').rstrip('\r') for line in open('replacement_word_list.txt')]

            replacement_words = {}
            for replacement_word in replacement_words_list :
                replacement_words[replacement_word.split(',')[0]] = replacement_word.split(',')[1]
            
            new_string = []
            for word in tweet.split():
                if replacement_words.get(word,None) is not None :
                    word = replacement_words[word]
                new_string.append(word)

            tweet = ' '.join(new_string)

            #stemming
            stem_factory = StemmerFactory()
            stemmer = stem_factory.create_stemmer()
            tweet = stemmer.stem(tweet)

            #remove stopwords
            stopword_factory = StopWordRemoverFactory()
            stopword = stopword_factory.create_stop_word_remover()
            tweet = stopword.remove(tweet)

            tweet_final.append(tweet)

        tweet_final = np.array(tweet_final)
        return tweet_final

    def fit_transform(self,X) :
        return self.transform(X)

class CustomPipeline():
	def __init__(self, predictor):
		self.predictor = predictor
	def fit(self,X,y):
		tweet_cleaner = TweetCleaner()
		count_vectorizer = CountVectorizer()
		tfidf_transformer = TfidfTransformer()
		used_predictor = self.predictor
		X = tweet_cleaner.fit_transform(X)
		X = count_vectorizer.fit_transform(X)
		X = tfidf_transformer.fit_transform(X)
		used_predictor.fit(X,y)
		
		self.tweet_cleaner = tweet_cleaner
		self.count_vectorizer = count_vectorizer
		self.tfidf_transformer = tfidf_transformer
		self.used_predictor = used_predictor
		
	def predict(self,X):
		
		X = self.tweet_cleaner.transform(X)
		X = self.count_vectorizer.transform(X)
		X = self.tfidf_transformer.transform(X)
		prediction = self.used_predictor.predict(X)
		return prediction
		


