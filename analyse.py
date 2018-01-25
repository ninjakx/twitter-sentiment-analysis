import tweepy
import pandas as pd
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import collections
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re, string
import nltk
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim.models import Word2Vec
from textblob import TextBlob
from gensim import corpora, models


df = read_csv('output.csv')
df.columns = ['username','retweets','favorites','text','mentions','hashtags','id','permalink']
df.to_csv('test_2.csv')
df = read_csv('test_2.csv')



################ TOP HASHTAGS #########################

#df.ix[0]
hashtags = []
h = []
#print(df["hashtags"])

for hs in df["hashtags"]: # Each entry may contain multiple hashtags. Split.
       if hs != hs:
           
           continue
       #print(len(hs))
       if len(hs)>2: 
           hashtags += hs.split(" ")


for i in range(1,len(hashtags),5):
    #print(i)
    #if (hashtags[i][0]) == '^':
    #    hashtags[i] = (hashtags[i].split('^'))[1]
        #print(hashtags[i])
    
    h.append(hashtags[i])

counter=collections.Counter(h)
#print(counter.keys)
count = 0
x1_val,y1_val = [],[]
for i,j in counter.most_common():
     count+=1
     if count!=11:
         x1_val.append(i)
         y1_val.append(j)
     else:
         break
         
plt.figure(figsize=(25,25))
plt.subplot(1, 2, 1)
plt.plot(range(10), y1_val, linestyle='-', marker='o')
#plt.subplot(121)
plt.xticks(range(10), x1_val)
plt.title('TOP HASHTAGS')

#Rotate labels by 90 degrees so you can see them
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

plt.gcf().subplots_adjust(bottom=0.25)


############## TOP USERS ##########################         

users = df["username"].tolist()
#print(users)
counter=collections.Counter(users)
#print(counter)
count = 0
x2_val,y2_val = [],[]
for i,j in counter.most_common():
     count+=1
     if count!=11:
         x2_val.append(i)
         y2_val.append(j)
     else:
         break

plt.subplot(1, 2, 2)
plt.plot(range(10), y2_val, linestyle='-', marker='o')

plt.xticks(range(10), x2_val)
plt.title('TOP USERS')
#Rotate labels by 90 degrees so you can see them
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)

plt.gcf().subplots_adjust(bottom=0.25)

plt.show()

################################ PROCESS TWEETS ########################

tweets_texts = df["text"].tolist()
stopwords=stopwords.words('english')
english_vocab = set(w.lower() for w in nltk.corpus.words.words()) # Remove repeated words

def process_text(tweet):
   # https://www.tutorialspoint.com/python/string_startswith.htm
   if tweet.startswith('@null'):
       return "[Tweet not available]"
   tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
   tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
   tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove punctuations like ' !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '

   twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
   tokens = twtok.tokenize(tweet)
   
   tokens = [i.lower() for i in tokens if i not in stopwords and len(i) > 2 and  
                                             i in english_vocab]
   return tokens
words = []
for tw in tweets_texts:
    words += process_text(tw)
#print(words)


################### FIND BIGRAM #######################

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words, 5)
finder.apply_freq_filter(5)
#print(finder.nbest(bigram_measures.likelihood_ratio, 10))



def get_tweet_sentiment(tweet):
    # create TextBlob object of passed tweet text
    analysis = TextBlob(tweet)
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


#################### PRE PROCESS TWEETS ####################

cleaned_tweets = []
for tw in tweets_texts:
    words = process_text(tw)
    cleaned_tweet = " ".join(w for w in words if len(w) > 2 and 
w.isalpha()) #Form sentences of processed words
    cleaned_tweets.append(cleaned_tweet)
df['CleanTweetText'] = cleaned_tweets
#print(df['CleanTweetText'])


################# SENTIMENTS ###################

Sentiment = []
pos_tweets = []
neg_tweets = []
neutral_tweets = []
for text in cleaned_tweets:
    senti = get_tweet_sentiment(text)
    Sentiment.append(senti)
    if senti == 'positive':
        pos_tweets.append(text)
    elif senti == 'negative':
        neg_tweets.append(text)
    elif senti == 'neutral':
        neutral_tweets.append(text) 
#print(Sentiment)
#print(len(Sentiment))

idx = 4
df.insert(loc=idx, column='sentiment', value=Sentiment)
df.to_csv('test_2.csv')


print("Positive tweets percentage: {} %".format(100*len(pos_tweets)/len(Sentiment)))
print("Negative tweets percentage: {} %".format(100*len(neg_tweets)/len(Sentiment)))
print("Neutral tweets percentage: {} % \
        ".format(100*(len(Sentiment) - len(neg_tweets) - len(pos_tweets))/len(Sentiment)))
 
# printing first 5 positive tweets
print("\n\nPositive tweets:")
for tweet in pos_tweets[:10]:
    print(tweet)
 
# printing first 5 negative tweets
print("\n\nNegative tweets:")
for tweet in neg_tweets[:10]:
    print(tweet)



################## CLUSTER THE WORDS ##############################

from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))  
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_tweets)  
feature_names = tfidf_vectorizer.get_feature_names() # num phrases  

from sklearn.cluster import KMeans  
num_clusters = 3
km = KMeans(n_clusters=num_clusters)  
km.fit(tfidf_matrix)  
clusters = km.labels_.tolist()  
df['ClusterID'] = clusters  
print(df['ClusterID'].value_counts())

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster {} : Words :".format(i))
    for ind in order_centroids[i, :10]: 
        print(' %s' % feature_names[ind])



################## CLUSTERING THE TWEETS USING LDA ########################

words_list = []
stop = stopwords
#stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
texts = [text for text in cleaned_tweets if len(text) > 2]
doc_clean = [clean(doc).split() for doc in texts]
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
ldamodel = models.ldamodel.LdaModel(doc_term_matrix, num_topics=6, id2word = 
dictionary, passes=5)
for topic in ldamodel.show_topics(num_topics=6, formatted=False, num_words=15):
    for (w,val) in topic[1]:
        words_list.append(w)
    
    #print(list(w for (w, val) in topic[1]))
    print("Topic {}: Words: ".format(topic[0]))
    topicwords = [w for (w, val) in topic[1]]
    print(topicwords)
#print(words_list)


###################### CLUSTERING THE TWEETS USING DOC2VEC & K-MEANS ####################
              ######### TRAIN THE MODEL ##########


import gensim
from gensim.models.doc2vec import TaggedDocument
taggeddocs = []
tag2tweetmap = {}
token_count = sum([len(sentence) for sentence in taggeddocs])
for index,i in enumerate(cleaned_tweets):
    if len(i) > 2: # Non empty tweets
        tag = u'SENT_{:d}'.format(index)
        sentence = TaggedDocument(words=gensim.utils.to_unicode(i).split(), 
tags=[tag])
        tag2tweetmap[tag] = i
        taggeddocs.append(sentence)
model = gensim.models.Doc2Vec(taggeddocs, dm=0, alpha=0.025, size=20, 
min_alpha=0.025, min_count=0)
for epoch in range(60):
    if epoch % 20 == 0:
        print('Now training epoch %s' % epoch)
    model.train(taggeddocs, total_examples = token_count,epochs = model.iter)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay



           ######### TOPIC WISE WORDS #########

from sklearn.cluster import KMeans
dataSet = model.wv.syn0
kmeansClustering = KMeans(n_clusters=6)
centroidIndx = kmeansClustering.fit_predict(dataSet)
topic2wordsmap = {}
for i, val in enumerate(dataSet):
    tag = str(model.docvecs.index_to_doctag(i))
    if not tag.startswith('SENT_'):
        continue
    topic = centroidIndx[i]
    if topic in topic2wordsmap.keys():
         #print(tag2tweetmap[tag])
         for w in (tag2tweetmap[tag].split()):
             topic2wordsmap[topic].append(w)
    else:
          topic2wordsmap[topic] = []
for i in topic2wordsmap:
    words = topic2wordsmap[i]
    print("Topic {} has words {}".format(i, words[:15]))



######################## WORDS CLOUD #######################

d = path.dirname(__file__)

twitter_mask = np.array(Image.open(path.join(d, "twitter_mask.png")))
stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=400, mask=twitter_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(" ".join(words_list))

# store to file
wc.to_file(path.join(d, "word_cloud.jpg"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

