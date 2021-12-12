#######################################################################
# Datalab
# Schadcode für Webserver
# Gruppe: Hex Haxors
# Zaid Askari & Oussema Ben Taarit
#######################################################################
# Improting the initial necessary libraries

import numpy as np
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator

import seedir as sd
from io import BytesIO
import os
import re
import urlextract 
import nltk
#######################################################################
# Define the variables 
labels = []

#######################################################################
# Extract the text from each document

train_zip = zipfile.ZipFile("3_Schadcode_in_software/train.zip")
names = train_zip.namelist() 
# removing the labels file
names = names[:-1]
contents = []
#print(names)
url_extractor = urlextract.URLExtract()
stemmer = nltk.PorterStemmer()
for name in names:
    #print(name)
    tokens = name.split(".")
    labels.append(int(tokens[1]))

for i in range(0, 10): #len(names))
    #print(type(name))
    content = train_zip.read(names[i]).decode("latin-1")
    #print(test.decode("ISO-8859-1"))
    
    urls = list(set(url_extractor.find_urls(content)))
    urls.sort(key=lambda url: len(url), reverse=True)
    for url in urls:
        content = content.replace(url, " URL ")
        #print(url)
    content = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' NUMBER ', content)
    contents.append(content)

print(len(contents[0]))
print(len(contents[1]))
#print(type(contents[0]))
# print(type(names[0]))
#content = [train_zip.read(names[0]).decode("latin-1")]
#print(content)
vectorizer = CountVectorizer(lowercase = True, min_df = 0.50, analyzer = 'word', ngram_range = (1, 1)) #max_df = 0.5
content_updated = vectorizer.fit_transform(contents)
print(vectorizer.vocabulary_)
print(type(content_updated[0]))


print(content_updated[0].shape)
print(content_updated[1].shape)



train_zip.close()