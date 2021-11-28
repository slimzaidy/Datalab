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

train_zip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip")
names_big_files = train_zip.namelist() # There are 6301 docx files after removing the labels file in train.zip
# removing the labels file
names_big_files = names_big_files[:-1]
names_big_files = names_big_files[:100]  

print(len(names_big_files))
for name in names_big_files:
    print(name)
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
import csv

f_csv = open("doc_text.csv", "w+", newline ='', encoding="utf-8")
writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

for name_big_file, label, text in zip(names_big_files, labels, doc_text):
    x = name_big_file + ";" + str(label) + ";" +  text  
    writer.writerow([x])

f_csv.close()