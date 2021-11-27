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
X = []

#########################################################################################
# Data preprocessing 


myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train/data/docx-2016-07/aaduuijtoewjqttc.0") 
#myzip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip") 


names = myzip.namelist()
#names = names[-1]

for filename in names:
    if filename == 'word/document.xml':
        #X.append('\n'.join([line.decode('latin-1').lower() for line in myzip.read(filename).splitlines()]))
        #labels.append(int(filename[-1]))
        #X = ['\n'.join (filter(lambda line: len(line) > 0, mail.splitlines())) for mail in X]
        # Read each line in the file, readlines() returns a list of lines
        content = filename.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        #bs_content = bs(content, "lxml")
        print(X)

#print(X)

# # file_to_test = names['word/document.xml']
# Index_Number_For_doc_xml = names.index('word/document.xml')
# print(Index_Number_For_doc_xml)    

# xml_doc = myzip.open(names[Index_Number_For_doc_xml])
# content = xml_doc.read()


soup = BeautifulSoup(X, 'lxml')
print(soup.prettify())

# wow = soup.find_all('w:p')
#print(wow)
#print(soup.prettify())
# print(soup.get_text())

myzip.close()


# #########################################################################################
# # Extract the text from each document

# train_zip = zipfile.ZipFile("2_Schadcode in Dokumenten/train.zip")
# names_big_files = train_zip.namelist()
# # removing the labels file
# names_big_files = names_big_files[:-1] 
# names_big_files = names_big_files[40:]

# for name in names_big_files:
#     print(name)
#     tokens = name.split(".")
#     labels.append(int(tokens[1]))

#     zfiledata = BytesIO(train_zip.read(name))
    
#     labeled_files_zip = zipfile.ZipFile(zfiledata)
#     sub_names = labeled_files_zip.namelist()
#     Index_Number_For_doc_xml = sub_names.index('word/document.xml')
#     xml_doc = labeled_files_zip.open(sub_names[Index_Number_For_doc_xml])
#     content = xml_doc.read()
#     soup = BeautifulSoup(content, 'lxml')
#     text = soup.get_text()
#     #text = text.encode("utf8")
#     doc_text.append(text) #(str(text))

#     labeled_files_zip.close()

# train_zip.close()

# print(labels[0])
# #print(str(doc_text[0]))

# #########################################################################################
# # Save the text and labels with the names of the file to a csv
# import csv

# f_csv = open("doc_text.csv", "w+", newline ='')
# writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

# for name_big_file, label, text in zip(names_big_files, labels, doc_text):
#     x = name_big_file + ";" + str(label) + ";" + str(text.encode("utf8")) #str(text.encode("utf8"))
#     writer.writerow([x])

# f_csv.close()