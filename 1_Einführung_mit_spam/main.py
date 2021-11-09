#########################################################################################
# Resources

# https://www.sec.tu-bs.de/teaching/ws21/datalab/videos/datalab-01-spam-2.mp4

# Improting all necessary libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os, shutil
from datetime import datetime
from pickle import load, dump
import zipfile

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.base import TransformerMixin, BaseEstimator

from keras.models import Sequential, save_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

#print(emails[0])

for name in names:
    tokens = name.split(".")
    labels.append(int(tokens[1]))

#print(len(labels))

#########################################################################################
# Feature engineering, try also: 1. n-gram | 2. bag of words

X = np.zeros((16662, 3))
#print(X.shape)

for i, email in enumerate(emails): 
    X[i, 0] = len(email)            # mean length: 1476.67 | max len = 28445 | min len = 132
    X[i, 1] = len(email.split())    # mean length: 300.24 | max len = 6132 |
    X[i, 2] = 'click' in str(email)

Y = np.array(labels)

#########################################################################################
# Prepeare the data and build the model

X_train, X_test, y_train, y_test = train_test_split(
    X[0:16662],
    Y[0:16662],
    test_size=0.2,
    shuffle=True
)

model = SVC(C = 1000) # C = 1
model.fit(X_train, y_train)
yhat = model.predict(X_test)
#print(np.mean(y_test))

print(np.mean(yhat == y_test))
print("balanced accuracy score = {}".format(balanced_accuracy_score(y_test, yhat)))

z.close()

###################################>>>>>$$$$$$$$$$$$$$$$<<<<<#############################
# open the other test data zip file, preproces and perform predictions on it

emails = []
labels = []
z = zipfile.ZipFile("1_Einführung_mit_spam/1_SPAM_ERKENNUNG_MIT_MASCHINELLEM_LERNEN/spam1-test.zip")
names = z.namelist()

for name in names:
    email = z.read(name)
    emails.append(email)

#print(len(emails))

X = np.zeros((15051, 3))
#print(X.shape)

for i, email in enumerate(emails): 
    X[i, 0] = len(email)            
    X[i, 1] = len(email.split())    
    X[i, 2] = 'click' in str(email)


yhat = model.predict(X_test)

#########################################################################################
# Save the predictions with the names of the emails in a new file
f = open("spam1-test.pred", "w")
for name, pred in zip(names, yhat):
    #print("%s;%d" % (name, pred))
    f.write("%s;%d" % (name, pred))

f.close()
z.close()
