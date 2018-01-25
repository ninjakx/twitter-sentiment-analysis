import got, codecs
#import pandas as pd
import numpy as np
import csv
import fileinput
import os

#np.savetxt('output.csv', ('username','retweets','favorites','text','geo','mentions','hashtags','id','permalink'), delimiter=',')
#df = pd.DataFrame(columns=['username','retweets','favorites','text','geo','mentions','hashtags','id','permalink'])
i=-1
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('india').setSince("2017-12-25").setUntil("2018-01-25").setMaxTweets(1100)


try:
    os.remove("output.csv")


except OSError:
    pass

#for line in fileinput.input(files=['output.csv'], inplace=True):
#    if fileinput.isfirstline():
#        print 'username,;retweets,;favorites,;text,;geo,;mentions,;hashtags,;id,;permalink'
#    print line,

def streamTweets(tweets):
   for t in tweets:
      #obj = {"user": t.username, "retweets": t.retweets, "favorites":  
      #      t.favorites, "text":t.text,"geo": t.geo,"mentions": 
      #      t.mentions, "hashtags": t.hashtags,"id": t.id,
      #      "permalink": t.permalink,}
      #print(obj)
      #np.savetxt('output.csv', (t.username,t.retweets,t.favorites,t.text,t.geo,t.mentions,t.hashtags,t.id,t.permalink), delimiter=',')
      '''mylist = [t.username.encode('utf-8'),'^'+ str(t.retweets),'^'+ str(t.favorites),'^'+t.text.encode('utf-8'),'^'+t.geo,'^'+t.mentions.encode('utf-8'),'^'+t.hashtags,'^'+ str(t.id),'^'+ t.permalink.encode('utf-8')]'''

      mylist = [t.username.encode('utf-8'),str(t.retweets),str(t.favorites),t.text.encode('utf-8'),t.mentions.encode('utf-8'),t.hashtags,str(t.id),t.permalink.encode('utf-8')]

      #df.loc[i] = [t.username, t.retweets, t.favorites,t.text,t.geo,t.mentions,t.hashtags, t.id, t.permalink,]
      with open('output.csv', 'a') as out:
          writer = csv.writer(out)
          writer.writerow(mylist)
#df.to_csv("output.csv")

got.manager.TweetManager.getTweets(tweetCriteria, streamTweets)





