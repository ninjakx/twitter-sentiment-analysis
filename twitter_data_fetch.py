import tweepy
import pandas as pd
import csv
import re

#import twitter apps configaration file from config.py file
from settings import *

#Authentication using keys & accesstoken
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

df = pd.DataFrame(columns=['hashtags','user','date','retweets','favorite','geo','id','text'])
i=-1
for tweet in tweepy.Cursor(api.search,
                           q="india",
                           since="2018-01-01",
                           until="2018-01-24",
                           lang="en").items():
    i+=1
    if (i!=900):
        df.loc[i] = [tweet.entities.get('hashtags'),tweet.author.screen_name,tweet.created_at,tweet.retweet_count,tweet.favorite_count,tweet.geo,tweet.id_str, tweet.text.encode("utf-8")]
    else:
        break
df.to_csv("output.csv")
