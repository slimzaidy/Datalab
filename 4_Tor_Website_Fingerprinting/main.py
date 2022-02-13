#########################################################################################
# Datalab
# Tor Website Fingerprinting - Closed World
# Gruppe: Hex Haxors
# Zaid Askari & Oussama Ben Taarit
#########################################################################################

from copy import Error
import os
from scapy import data
from scapy.utils import RawPcapReader
from scapy.all import *
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import csv
# https://www.youtube.com/watch?v=VdHEmMXeosU

path = "C:/Users/askar/OneDrive/Dokumente/Meine Dateien/Uni/IST/Datalab/Datalab/4_Tor_Website_Fingerprinting/train_data/"
path_test = "C:/Users/askar/OneDrive/Dokumente/Meine Dateien/Uni/IST/Datalab/Datalab/4_Tor_Website_Fingerprinting/test_data/"

dataframe = pd.DataFrame()
count_list = []
labels = []
sent_packets = []
rcv_packets = []
avg_packet_length = []
dataframe_test = pd.DataFrame()
count_list_test = []
file_names_test = []
sent_packets_test = []
rcv_packets_test = []
avg_packet_length_test = []
NOMINAL_TRAFO = RobustScaler()
CAT_TRAFO = OrdinalEncoder()
MODEL = 0

""" 
Train _ Create a dataframe of packet count and labels and export csv
"""
def load_pcap_train_and_save():
    for file in os.listdir(path):
        if file.endswith(".pcap"):
            new_path = os.path.join(path, file)
            traffic = rdpcap(new_path)
            label = file.split("_")[0]
            packet_count = len(traffic)
            labels.append(label)
            count_list.append(packet_count)
            sent = 0
            rcv = 0
            packet_lengths_current_file = []
            for packet in traffic:
                
                try:
                    src = packet[IP].src
                    length = packet[IP].len
                    packet_lengths_current_file.append(length)
                    if src == "134.169.109.25": 
                        sent += 1 
                    else: 
                        rcv += 1
                    #print(length)
                except IndexError as e:
                    pass

            #print(f'{label}: {sent} : {rcv}')
            avg_packet_length.append(int(np.mean(packet_lengths_current_file)))
            sent_packets.append(sent)
            rcv_packets.append(rcv)
            
    #print(f'len(count_list) and {len(sent_packets)} and {len(rcv_packets)}')
    dataframe["count"] = count_list
    dataframe["sent_packets"] = sent_packets
    dataframe["rcv_packets"] = rcv_packets
    dataframe["avg_packet_length"] = avg_packet_length
    dataframe["labels"] = labels

    print(dataframe)
    dataframe.to_csv("df_1_train_pre.csv")

""" Import dataframe and preprocess """
def preprocess_csv_train_and_save():
    preprocessed_df = pd.read_csv('df_1_train_pre.csv', index_col=0)

    postprocessed_df = preprocessed_df.copy() #pd.DataFrame()
    #print(postprocessed_df.columns[:-1])
    NOMINAL_TRAFO = NOMINAL_TRAFO.fit(postprocessed_df[postprocessed_df.columns[:-1]]) #postprocessed_df[["count"]]
    CAT_TRAFO = CAT_TRAFO.fit(postprocessed_df[["labels"]])

    postprocessed_df[postprocessed_df.columns[:-1]] = NOMINAL_TRAFO.transform(postprocessed_df[postprocessed_df.columns[:-1]])
    postprocessed_df["labels"] = CAT_TRAFO.transform(postprocessed_df[["labels"]])

    #print(postprocessed_df)

    postprocessed_df.to_csv("df_1_train_post.csv")

""" Import df and create model """
def import_csv_and_create_model():
    df_to_train = pd.read_csv('df_1_train_post.csv', index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(
        df_to_train[df_to_train.columns[:-1]],
        df_to_train["labels"],
        test_size=0.1,
        shuffle=True
    )

    #print(type(X_train.values)) # Must be a numpy array
    #print(type(y_train.values)) # Must be a numpy array

    MODEL = RandomForestClassifier()
    MODEL.fit(X_train.values, y_train.values.ravel())
    y_hat = MODEL.predict(X_test.values)

    accuracy = accuracy_score(y_test.values.ravel(), y_hat)
    #print(accuracy)

    df_to_train["labels"] = CAT_TRAFO.inverse_transform(df_to_train[["labels"]])

    return df_to_train

""" Prediction """


""" 
Test _ Create a dataframe of packet count and labels and export csv 
"""

def load_pcap_train_and_save():
    for file in os.listdir(path_test):
        if file.endswith(".pcap"):
            new_path = os.path.join(path_test, file)
            file_names_test.append(file)
            traffic = rdpcap(new_path)
            packet_count = len(traffic)
            count_list_test.append(packet_count)
            sent = 0
            rcv = 0
            packet_lengths_current_file = []
            for packet in traffic:
                
                try:
                    src = packet[IP].src
                    length = packet[IP].len
                    packet_lengths_current_file.append(length)
                    if src == "134.169.109.25": 
                        sent += 1 
                    else: 
                        rcv += 1
                    #print(length)
                except IndexError as e:
                    pass

            #print(f'{label}: {sent} : {rcv}')
            avg_packet_length_test.append(int(np.mean(packet_lengths_current_file)))
            sent_packets_test.append(sent)
            rcv_packets_test.append(rcv)
            
    #print(f'len(count_list_test) and {len(sent_packets_test)} and {len(rcv_packets_test)}')
    dataframe_test["count"] = count_list_test
    dataframe_test["sent_packets"] = sent_packets_test
    dataframe_test["rcv_packets"] = rcv_packets_test
    dataframe_test["avg_packet_length"] = avg_packet_length_test
    #dataframe_test["labels"] = labels

    #print(dataframe_test)
    dataframe_test.to_csv("df_1_test_pre.csv")

""" Test _ Preprocess """
def preprocess_csv_test_pred_save():
    preprocessed_df = pd.read_csv('df_1_test_pre.csv', index_col=0)


    postprocessed_df = preprocessed_df.copy() #pd.DataFrame()

    postprocessed_df[postprocessed_df.columns[:]] = NOMINAL_TRAFO.transform(postprocessed_df[postprocessed_df.columns[:]])


    postprocessed_df.to_csv("df_1_test_post.csv")

    """ Test _ Perform prediction """

    df_to_predict = pd.read_csv('df_1_test_post.csv', index_col=0)

    print(type(df_to_predict))

    y_predicted = MODEL.predict(df_to_predict.values)

    #print(y_predicted)

    df_to_predict["labels"] = CAT_TRAFO.inverse_transform(y_predicted.reshape(-1, 1)) #df_to_predict[["labels"]]

    #print(df_to_predict)


    """ Save to outout csv file """

    file_names = []
    for file in os.listdir(path_test):
        if file.endswith(".pcap"):
            file_names.append(file)


    f_csv = open("output_tor.csv", "w+", newline ='')
    writer = csv.writer(f_csv, quoting=csv.QUOTE_ALL) 

    for name, pred in zip(file_names, df_to_predict["labels"]):
        x = name + ";" + str(pred)
        writer.writerow([x])

    f_csv.close()

if __name__ == "__main__":
    load_pcap_train_and_save()
    preprocess_csv_train_and_save()
    import_csv_and_create_model()
    load_pcap_train_and_save()
    preprocess_csv_test_pred_save()