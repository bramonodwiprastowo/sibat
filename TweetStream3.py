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
import mysql.connector
from mysql.connector import Error
from mysql.connector import errorcode
from urllib3.exceptions import ProtocolError

from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

from library import TweetCleaner
from library import CustomPipeline


os.environ['TZ'] = 'Asia/Jakarta'

consumer_key = "UJ4SSjFzZ6P4NhBlGvCi3hJ3t"
consumer_secret = "LNZ0NcLVNeLz9thgbMa22vECP7NTnZjOtavIgJHjUkWfouyKuE"
access_token = "2996062224-tU0XoN9LESsQEmF1RF1bTGxl7yYjfvwH4PbiEdY"
access_secret = "G7QJUv7xmGHqBaTbRXSUFgpVJ4BGHheRQizbjN9itCmSY"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
	
SVM_classifier1 = pickle.load(open('svm1.sav','rb'))
SVM_classifier2 = pickle.load(open('svm2.sav','rb'))

class TwitterStreamer(StreamListener):
    def __init__(self):
        super(TwitterStreamer, self).__init__()
        
        print('\nTweets:')
        with open(os.path.dirname(__file__) + 'classified_tweets.txt', 'a') as f:
            f.write('\nTweets:')

    def on_data(self, data):
        try:
            tweet = json.loads(data)['text'].replace('\n', ' ')
            print(tweet)
            tweet = np.array([[tweet]])
            user_id = json.loads(data)['user']['id']#.get('id')
            print(user_id)
            if (user_id == 108543358 or user_id == 346303384 or user_id == 69183155 or user_id == 23343960 or user_id == 759692754985242625 or user_id == 18129942 or user_id == 4187275698 or user_id == 47596019) :
                prediction1 = SVM_classifier1.predict(tweet)
                result1 = prediction1[0]
                print(result1)
                """
                # if self.classifier.naive_bayes_classify(tweet) is 'traffic' or \
                # self.classifier.svm_classify(tweet) is 'traffic' or \
                # self.classifier.decision_tree_classify(tweet) is 'traffic':
                #nb_result = str(self.classifier.naive_bayes_classify(tweet))
                nb_result = str(self.classifier.naive_bayes_classify(tweet))
                #dt_result = str(self.classifier.decision_tree_classify(tweet))

                if sys.argv[1] == "dev":
                    print('| ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        '\t| ' + nb_result,
                        '\t| ' + tweet)
                    with open(os.path.dirname(__file__) + 'classified_tweets.txt', 'a') as f:
                        f.write('\n| ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                                '\t| ' + nb_result +
                                '\t| ' + tweet)
                    with open(os.path.dirname(__file__) + 'classified_tweets.csv', 'a') as f:
                        f.write('"' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                                '","' + nb_result +
                                '","' + tweet + '"\n')
                                
                """
                if result1 == "gempatsunami":
                    #ts = datetime.datetime.strftime(datetime.datetime.strptime(json.loads(data)['created_at'], 
                    #'%a %b %d %H:%M:%S +0000 %Y') + datetime.timedelta(hours=7), '%Y-%m-%d %H:%M:%S')
                    prediction2 = SVM_classifier2.predict(tweet)
                    result2 = prediction2[0]
                    print(result2)
                    if result2 == "gempa" :
                        con = mysql.connector.connect(host='localhost', database='sibatugm_berita', user='root', password='QB13askl#!09')
                        cur = con.cursor()
                        add_tweet = (
                        "INSERT INTO nontsunami (tweet_id) VALUES(%s)")
                        tweet_data = (
                            json.loads(data)['id_str'],
                        )
                        cur.execute(add_tweet, tweet_data)
                        con.commit()

                        cur.close()
                    
                    if result2 == "tsunami":
                        #ts = datetime.datetime.strftime(datetime.datetime.strptime(json.loads(data)['created_at'], 
                            #'%a %b %d %H:%M:%S +0000 %Y') + datetime.timedelta(hours=7), '%Y-%m-%d %H:%M:%S')

                        con = mysql.connector.connect(host='localhost', database='sibatugm_berita', user='root', password='QB13askl#!09')
                        cur = con.cursor()
                        add_tweet = (
                        "INSERT INTO tsunami(tweet_id) VALUES(%s)")
                        tweet_data = (
                            json.loads(data)['id_str'],
                        )
                        cur.execute(add_tweet, tweet_data)
                        con.commit()

                        cur.close()


                    if result2 == "lainnya":
                        #ts = datetime.datetime.strftime(datetime.datetime.strptime(json.loads(data)['created_at'], 
                            #'%a %b %d %H:%M:%S +0000 %Y') + datetime.timedelta(hours=7), '%Y-%m-%d %H:%M:%S')

                        con = mysql.connector.connect(host='localhost', database='sibatugm_berita', user='root', password='QB13askl#!09')
                        cur = con.cursor()
                        add_tweet = (
                        "INSERT INTO lainnya(tweet_id) VALUES(%s)")
                        tweet_data = (
                            json.loads(data)['id_str'],
                        )
                        cur.execute(add_tweet, tweet_data)
                        con.commit()

                        cur.close()
            
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, TwitterStreamer())
# keywords = [line.rstrip('\n') for line in open(os.path.dirname(__file__) + 'name_list.txt')]
users = ['108543358', '346303384', '69183155', '23343960', '759692754985242625', '18129942', '4187275698', '47596019']
#keywords = ['tsunami','gempa']
while True :
    try :
        twitter_stream.filter(follow=users, is_async=True) #tracks='keyword'
    except (ProtocolError, AttributeError) :
        continue

