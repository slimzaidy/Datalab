
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
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.base import TransformerMixin, BaseEstimator

from keras.models import Sequential, save_model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


emails = []
labels = []
z = zipfile.ZipFile("C:/Users/askar/OneDrive/Dokumente/Meine Dateien/Uni/IST/Datalab/Datalab/1_Einführung_mit_spam/1_SPAM_ERKENNUNG_MIT_MASCHINELLEM_LERNEN/spam1-train.zip")
names = z.namelist()
print(names)