#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:31:51 2018

@author: shivaniramesh
"""

from collections import Counter
import nltk
import pandas as pd
from textblob import TextBlob
import string


d=[]

#read it in and use eval to convert each line to a dictionary

data=open("/Users/shivaniramesh/Desktop/FALL 18/DATA SCIENCE/cwctweets123.txt")

data
for line in data:
    d.append(eval(line))
    print(d)
#then append it to a list
#there should be ten dictionaries in the list
    
print(len(d))

#Display the first element of d - check its length as well
print('Element 1 on Display:',d[0])

print('length of first element:',len(d[0]))


hlist=Counter()
text_collection = ""

list_senti = []

#Get all 'statuses' using key 'statuses'
#use each status to get text - text is a key
#get the locations from where the tweets were sent, if available
for item in d:
    print(item['statuses'])
    for stat in item['statuses']:
        dict_senti = {}
        
        print(stat['user'])
        dict_senti['user'] = stat['user']['name']
        
        #for loc in stat['user']:
        print(stat['user']['location'])
        dict_senti['loc'] = stat['user']['location']
            
        #for text in stat['text']:
           
        dict_senti['text'] = stat['text']
        print(dict_senti['text'])
        text_collection += dict_senti['text'].lower() + ""
            
        
        for hashtag in stat['entities']['hashtags']:
            print(hashtag['text'])
            hlist[hashtag['text']]+=1
            
        dict_senti["senti_score"] = TextBlob(dict_senti['text']).sentiment
        list_senti.append(dict_senti)

#Find the top 5 hashtags used in the tweets - hint look at 'entities'
        
print(hlist.most_common(5))            

print(text_collection)



###



pun=string.punctuation
dig=string.digits

dig
pun+= "\n\t\r"
table = str.maketrans(pun, len(pun) * " ")

text_collection=text_collection.translate(table)

table = str.maketrans(dig, len(dig) * " ")
text_collection=text_collection.translate(table)

stopwords = nltk.corpus.stopwords.words("english")
stopwords.append("http")
stopwords.append("https")
stopwords.append("co")
stopwords.append("rt")
text_collection_list= text_collection.split()
list_words = [word for word in text_collection_list if word not in stopwords and len(word) >=3]

#Create a frequency plot of the top 25 words
#Display the total sentiment score for each tweet 
freq_dist = nltk.FreqDist(list_words)
freq_dist.plot(25)

print("Sentiment Analysis:")
pd.DataFrame(list_senti, columns = list_senti[0].keys()) 
