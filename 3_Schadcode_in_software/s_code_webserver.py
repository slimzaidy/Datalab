#######################################################################
# Datalab
# Schadcode für Webserver
# Gruppe: Hex Haxors
# Zaid Askari & Oussema Ben Taarit
#######################################################################
# Improting the initial necessary libraries

import numpy as np
import zipfile
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

import seedir as sd
from io import BytesIO
import os
import re
import urlextract 
import nltk
from datetime import datetime
#######################################################################
# Define the variables 
labels = []
file_names = []
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
time_now = datetime.now().strftime("%H_%M_%S")

for name in names:
    #print(name)
    tokens = name.split(".")
    file_names.append(tokens[0])
    labels.append(int(tokens[1]))


index_of_last_file = 10
labels_temp = labels[:index_of_last_file]
for i in range(0, index_of_last_file): #len(names)
    content = train_zip.read(names[i]).decode("latin-1")
    urls = list(set(url_extractor.find_urls(content)))
    urls.sort(key=lambda url: len(url), reverse=True)
    for url in urls:
        content = content.replace(url, " URL ")
    content = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' NUMBER ', content)
    contents.append(content)

print(len(contents[0]))
print(len(contents[1]))
print(type(contents[0]))
# print(type(names[0]))
#content = [train_zip.read(names[0]).decode("latin-1")]
#print(content)
vectorizer = CountVectorizer(lowercase = True, max_df = 0.25, analyzer = 'word', ngram_range = (1, 1)) #max_df = 0.5 #max_features=1000
content_updated = vectorizer.fit_transform(contents)
#print(vectorizer.vocabulary_)
#print(type(content_updated[0]))

print(content_updated[0].shape)
print(content_updated[1].shape)
print(content_updated.shape)
train_zip.close()

# model = RandomForestClassifier(n_estimators=1000)
# model.fit(content_updated[:-3], labels_temp[:-3])
# y_pred = model.predict(content_updated[-3:])
# print(y_pred)
####### Clustering ###########

df = pd.DataFrame(content_updated.toarray(), columns=vectorizer.get_feature_names())
print(df)

n_clusters = 4
kmeans = KMeans(n_clusters) 
y_pred = kmeans.fit_predict(content_updated)
u_labels = np.unique(y_pred)
print(u_labels)


print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(n_clusters):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
    print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
print(kmeans.labels_)
print(kmeans.cluster_centers_)


n_components = 2
pca = PCA(n_components)
table = pd.DataFrame()
table['names'] = file_names[:index_of_last_file]
table['pred'] = y_pred
table['label'] = labels_temp
table['x'] = pca.fit_transform(df)[:, 0]
table['y'] = pca.fit_transform(df)[:, 1]
table['ARS_score'] = adjusted_rand_score(table['label'].values, table['pred'].values)
print(table)

np.random.seed(19680801)
colors = np.random.rand(n_components)
plt.scatter(table['x'], table['y'],  c = kmeans.labels_.astype(float), alpha=0.5) 
plt.savefig('results/foo' + str(n_clusters) + '_' + time_now + '.png')
plt.show()
plt.close()


table.to_csv('results/out_' + str(n_clusters) + '_' + time_now + '.csv')



