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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from sklearn.base import TransformerMixin, BaseEstimator

#########################################################################################
# Define the variables 
emails = []
labels = []

#########################################################################################
# Data preprocessing 
z = zipfile.ZipFile("1_Einführung_mit_spam/1_SPAM_ERKENNUNG_MIT_MASCHINELLEM_LERNEN/spam1-train.zip")
names = z.namelist()
names = names[:-1]
#print(len(names))

for name in names:
    email = z.read(name)
    emails.append(email)

for name in names:
    tokens = name.split(".")
    labels.append(int(tokens[1]))

#########################################################################################
# Prepeare the data and build the model

Y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    emails[0:16662], 
    Y[0:16662], 
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
            text = str(email) or ""
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

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42) #max_iter=1000
print(X_train_transformed.shape)
print(y_train.shape)
log_clf.fit(X_train_transformed, y_train)
yhat = log_clf.predict(X_test_transformed)
print("Balanced accuracy score on the training data = {:.2f}%".format(100 *balanced_accuracy_score(y_test, yhat)))


z.close()
###########################################***********################***********##########################
# Predict on the test set

emails_testset = []
labels_testset = []
z_testset = zipfile.ZipFile("1_Einführung_mit_spam/1_SPAM_ERKENNUNG_MIT_MASCHINELLEM_LERNEN/spam1-test.zip")
names_testset = z_testset.namelist()

for name in names_testset:
    email = z_testset.read(name)
    emails_testset.append(email)

X_test_transformed_testset = preprocess_pipeline.transform(emails_testset)
y_pred = log_clf.predict(X_test_transformed_testset)

#############################################################################################################
# Save the predictions with the names of the emails_testset in a new file
import csv

f_csv = open("output.csv", "w+", newline ='')
writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

for name, pred in zip(names_testset, y_pred):
    x = name + ";" + str(pred)
    writer.writerow([x])

f_csv.close()
z_testset.close()