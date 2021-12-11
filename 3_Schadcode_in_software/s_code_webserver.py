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


from sklearn.metrics import balanced_accuracy_score
from sklearn.base import TransformerMixin, BaseEstimator

import seedir as sd
from io import BytesIO
import os

#######################################################################
# Define the variables 
labels = []

#######################################################################
# Extract the text from each document

train_zip = zipfile.ZipFile("3_Schadcode_in_software/train.zip")
names = train_zip.namelist() 
# removing the labels file
names = names[:-1]

#print(names)

for name in names:
    #print(name)
    tokens = name.split(".")
    labels.append(int(tokens[1]))

# for name in names:
#     email = train_zip.read(name)

content = train_zip.read(names[1])
print(str(content).splitlines())

# https://download.datalab.sec.tu-bs.de/03-clust/train/eeea94124bc446ee5abe2156572d658f0569f06a.zip