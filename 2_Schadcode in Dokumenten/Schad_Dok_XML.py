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
#########################################################################################
# Define the variables 
emails = []
labels = []
#########################################################################################
# Data preprocessing 


# #myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train/data/docx-2016-07/aabhxehabdfcfopa.0")
# myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train/data/docx-2016-07/acuinrpjavavgvog.1")

# names = myzip.namelist()

# #names = names[:-1]
# # file_to_test = names['word/document.xml']
# Index_Number_For_doc_xml = names.index('word/document.xml')
# print(Index_Number_For_doc_xml)    

# xml_doc = myzip.open(names[Index_Number_For_doc_xml])
# data = xml_doc.read()


# soup = BeautifulSoup(data, 'lxml')
# #print(soup.prettify())
# print(soup.get_text())

# myzip.close()


train_zip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip")
names_big_files = train_zip.namelist()
# removing the labels file
names_big_files = names_big_files[:-1] 

for name in names_big_files:
    tokens = name.split(".")
    labels.append(int(tokens[1]))


train_zip.close()