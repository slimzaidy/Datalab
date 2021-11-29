#########################################################################################
# Datalab
# Schadcode im DOCX-Format
# Gruppe: Hex Haxors
# Zaid Askari & Oussema Ben Taarit
#########################################################################################
# Improting the initial necessary libraries

import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator

from bs4 import BeautifulSoup
import seedir as sd
from io import BytesIO
import os
#########################################################################################
# Define the variables 
doc_text = []
labels = []

#########################################################################################
# Data preprocessing 


# myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train_old/data/docx-2016-07/aaduuijtoewjqttc.0")
#myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train_old/data/docx-2017-01/uclvhtuckhtprhgn.1")  

#myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip") 

#print(zipfile.is_zipfile("2_Schadcode in Dokumenten/train_old/data/docx-2016-07/aaduuijtoewjqttc.0"))
# names = myzip.namelist()
# #names = names[-1]

# for filename in names:
#     if filename == 'word/document.xml':

#         with myzip.open(filename, 'r') as f:
#             soup = BeautifulSoup(f, 'lxml')
#             #print(soup.prettify())
#             for el in soup.find_all('w:p'):
#                 print(el.text)


# myzip.close()


# #########################################################################################
# # Extract the text from each document
max_number_of_data = 3000
train_zip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip")
names_docx_files = train_zip.namelist() # There are 6301 docx files after removing the labels file in train.zip
# removing the labels file
names_docx_files = names_docx_files[:-1]
names_docx_files = names_docx_files[:max_number_of_data]  

print(len(names_docx_files))
for name in names_docx_files:
    #print(name)
    tokens = name.split(".")
    labels.append(int(tokens[1]))

    # The docx file
    text_content = ''
    zfiledata = BytesIO(train_zip.read(name))

    is_zip_condition = zipfile.is_zipfile("2_Schadcode in Dokumenten/train_old/" + name)
    
    if is_zip_condition:
        labeled_files_zip = zipfile.ZipFile(zfiledata)
        sub_names = labeled_files_zip.namelist()

        for sub_name in sub_names:
            if sub_name == 'word/document.xml':
                with labeled_files_zip.open(sub_name, 'r') as f:
                    soup = BeautifulSoup(f, 'lxml')
                    #print(soup.prettify())
                    for el in soup.find_all('w:p'): # Here we have many separate lines of type string, maybe add them all together into one string
                        if (len(el.text) > 0):
                            text_content= " ".join((text_content, el.text)) #text_content + ' ' + el.text
                            
        doc_text.append(text_content)
        labeled_files_zip.close()
    else: 
        #text_content= " ".join((text_content, "defect"))
        doc_text.append("defect")

train_zip.close()

print(len(labels))
print(len(doc_text))
# #print(str(doc_text[0]))

# #########################################################################################
# # Save the text and labels with the names of the file to a csv
# import csv

# f_csv = open("doc_text.csv", "w+", newline ='', encoding="utf-8")
# writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

# for name_big_file, label, text in zip(names_docx_files, labels, doc_text):
#     x = name_big_file + ";" + str(label) + ";" +  text  
#     writer.writerow([x])

# f_csv.close()

#########################################################################################
# Prepeare the data and build the model

Y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    doc_text[0:max_number_of_data], 
    labels[0:max_number_of_data], 
    test_size=0.1,
    shuffle=True
)
print(len(X_train), len(X_test), len(y_train), len(y_test))

############################################################################################################
# Perform stemming
try:
    import nltk
    stemmer = nltk.PorterStemmer()
except ImportError:
    print("Error: stemming requires the NLTK module.")
    stemmer = None

##############################################################################################################
# URL extract: replace URLs with the word "URL"
try:
    import urlextract 
    url_extractor = urlextract.URLExtract()
except ImportError:
    print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None

##############################################################################################################
# We add create a transformer that we will use to convert emails to word counters and we use it later in the pipeline

from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import Counter

class EmailToWordCounterTrafo(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

##############################################################################################################
# Convert word counts to vectors and we use it later in the pipeline

from scipy.sparse import csr_matrix

class WordCounterToVectorTrafo(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000): 
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

#############################################################################################################
# Build the model and perform the prediction

from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTrafo()),
    ("wordcount_to_vector", WordCounterToVectorTrafo()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

X_test_transformed = preprocess_pipeline.transform(X_test)

#log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42) #max_iter=1000
model = RandomForestClassifier(n_estimators=1000)
#print(X_train_transformed.shape)
#print(y_train.shape)
model.fit(X_train_transformed, y_train)
yhat = model.predict(X_test_transformed)
print("Balanced accuracy score on the training data = {:.2f}%".format(100 *balanced_accuracy_score(y_test, yhat)))

###########################################***********################***********##########################
# Predict on the test set

doc_text_testset = []


test_zip = zipfile.ZipFile("2_Schadcode in Dokumenten/test.zip")
names_docx_files_test = test_zip.namelist() # There are 6301 docx files after removing the labels file in test.zip
#names_docx_files_test = names_docx_files_test[:max_number_of_data]  

print(len(names_docx_files_test))

for name in names_docx_files_test:
    #print(name)
    # The docx file
    text_content = ''
    zfiledata = BytesIO(test_zip.read(name))

    is_zip_condition = zipfile.is_zipfile("2_Schadcode in Dokumenten/test_old/" + name)
    
    if is_zip_condition:
        try:
            labeled_files_zip = zipfile.ZipFile(zfiledata)
            sub_names = labeled_files_zip.namelist()

            for sub_name in sub_names:
                if sub_name == 'word/document.xml':
                    try: 
                        with labeled_files_zip.open(sub_name, 'r') as f:
                            soup = BeautifulSoup(f, 'lxml')
                            #print(soup.prettify())
                            for el in soup.find_all('w:p'): # Here we have many separate lines of type string, maybe add them all together into one string
                                if (len(el.text) > 0):
                                    text_content= " ".join((text_content, el.text)) #text_content + ' ' + el.text
                    except Exception as e:
                        text_content = "defect"              
            doc_text_testset.append(text_content)
            labeled_files_zip.close()
        except Exception as e:
            doc_text_testset.append("defect")
    else: 
        doc_text_testset.append("defect")

test_zip.close()

X_test_transformed_testset = preprocess_pipeline.transform(doc_text_testset)
y_pred = model.predict(X_test_transformed_testset)

#############################################################################################################
# Save the predictions with the names of the emails_testset in a new file
import csv

f_csv = open("output.csv", "w+", newline ='')
writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

for name, pred in zip(names_docx_files_test, y_pred):
    x = name + ";" + str(pred)
    writer.writerow([x])

f_csv.close()
