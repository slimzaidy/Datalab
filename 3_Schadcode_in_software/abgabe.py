#######################################################################
# Datalab
# Schadcode fuer Webserver
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
import csv
#######################################################################
# Define the variables 
labels = []
file_names = []

#######################################################################
# Extract the text from each document

train_zip = zipfile.ZipFile("train.zip")
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
    file_names.append(tokens[0])
    labels.append(int(tokens[1]))


index_of_last_file = len(names)
labels_temp = labels[:index_of_last_file]
for i in range(0, index_of_last_file): 
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
vectorizer = CountVectorizer(lowercase = True, analyzer = 'word', max_df = 0.25, ngram_range = (1, 1), max_features=10000) #max_df = 0.25 #max_features=1000# max_df = 0.25,
content_updated = vectorizer.fit_transform(contents)
#print(vectorizer.vocabulary_)
#print(type(content_updated[0]))

print(content_updated[0].shape)
print(content_updated[1].shape)
print(content_updated.shape)
train_zip.close()

####### Clustering ###########

df = pd.DataFrame(content_updated.toarray(), columns=vectorizer.get_feature_names())
#print(df)
pca_orig = PCA(6)
df_pca = pca_orig.fit_transform(df)
#print(pca_orig.fit_transform(df))
print(pca_orig.explained_variance_ratio_)

n_clusters = 4
kmeans = KMeans(n_clusters) 
y_pred = kmeans.fit_predict(df_pca)
#u_labels = np.unique(y_pred)
#print(u_labels)


n_components = 2
pca = PCA(n_components)
table = pd.DataFrame()
table['names'] = file_names[:index_of_last_file]
table['pred'] = y_pred
table['label'] = labels_temp
table['x'] = pca.fit_transform(df_pca)[:, 0]
table['y'] = pca.fit_transform(df_pca)[:, 1]
table['ARS_score'] = adjusted_rand_score(table['label'].values, table['pred'].values)
print(table)
time_now = datetime.now().strftime("%H_%M_%S")
np.random.seed(19680801)
colors = np.random.rand(n_components)
plt.scatter(table['x'], table['y'],  c = kmeans.labels_.astype(float), alpha=0.5) 
plt.savefig('results/foo' + str(n_clusters) + '_' + time_now + '.png')
#plt.show()
plt.close()

table.to_csv('results/out_new_' + str(n_clusters) + '_' + time_now + '.csv')

####################################### Testing ############################################

file_names_test = []

# Extract the text from each document

test_zip = zipfile.ZipFile("test.zip")
names_test = test_zip.namelist() 
# removing the labels file
contents_test = []

for name in names_test:
    #print(name)
    tokens = name.split(".")
    file_names_test.append(tokens[0])
    #labels.append(int(tokens[1]))


#
#index_of_last_file_test = len(names_test)
#labels_temp = labels[:index_of_last_file]
for names_test in file_names_test: #len(names)
    content = test_zip.read(names_test).decode("latin-1")
    urls = list(set(url_extractor.find_urls(content)))
    urls.sort(key=lambda url: len(url), reverse=True)
    for url in urls:
        content = content.replace(url, " URL ")
    content = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' NUMBER ', content)
    contents_test.append(content)

content_updated_test = vectorizer.transform(contents_test)
test_zip.close()

df_test = pd.DataFrame(content_updated_test.toarray(), columns=vectorizer.get_feature_names())
print(df_test)
####################################

df_pca = pca_orig.transform(df_test)
#print(pca_orig.fit_transform(df))
print(pca_orig.explained_variance_ratio_)

y_pred_test = kmeans.predict(df_pca)

time_now = datetime.now().strftime("%H_%M_%S")

table_test = pd.DataFrame()
table_test['names'] = file_names_test
table_test['pred'] = y_pred_test
#table['label'] = labels_temp
#n_components = 2
#pca_test_2 = PCA(n_components)
#table_test['x'] = pca_test_2.fit_transform(df_pca)[:, 0]
#table_test['y'] = pca_test_2.fit_transform(df_pca)[:, 1]
#table['ARS_score'] = adjusted_rand_score(table['label'].values, table['pred'].values)
#print(table_test)

#np.random.seed(19680801)
#colors = np.random.rand(n_components)
#plt.scatter(table_test['x'], table_test['y'], alpha=0.5) #c = kmeans.labels_.astype(float),
#plt.savefig('results_test/foo' + str(n_clusters) + '_' + time_now + '.png')
#plt.show()
#plt.close()

table_test.to_csv('results_test/output_' + str(n_clusters) + '_' + time_now + '.csv')

f_csv = open("output.csv", "w+", newline ='')
writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

for name, pred in zip(file_names_test, y_pred_test):
    x = name + ";" + str(pred)
    writer.writerow([x])

f_csv.close()



