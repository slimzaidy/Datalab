import os
import pyshark
from scapy import data
import helping_functions as hf
import datetime
import dpkt
from scapy.utils import RawPcapReader
from scapy.all import *
import sys
import os
import pandas as pd
# https://www.youtube.com/watch?v=VdHEmMXeosU

path = "C:/Users/askar/OneDrive/Dokumente/Meine Dateien/Uni/IST/Datalab/Datalab/4_Tor_Website_Fingerprinting/train_data/"


# with open(path, 'rb') as f: #encoding = "cp1252", errors="ignore"
#     print(f)
#     pcap = dpkt.pcap.Reader(f)


# labels = []

# string = "appspot.com_2.pcap"

# label = string.split("_")[0]
#print(label)

# The number of files in the directory
#print([name for name in os.listdir(path) if name.endswith(".pcap")])

# for file in os.listdir(path):
#     if file.endswith(".pcap"):
#         label = file.split("_")[0]
#         labels.append(label)
#         print(file)
#         #data = hf.read_pcap_file(file, path)
#         with open(os.path.join(path, file)) as data:
#             print(data)
#             pcap = dpkt.pcap.Reader(data)
#             for timestamp, buf in pcap:
#                 print('Timestamp: ', str(datetime.datetime.utcfromtimestamp(timestamp)))

#import dpkt
#import socket


# def printPcap(pcap):
# 	for (ts,buf) in pcap:
# 		try:
# 			eth = dpkt.ethernet.Ethernet(buf)
# 			ip = eth.data
# 			# read the source IP in src
# 			src = socket.inet_ntoa(ip.src)
# 			# read the destination IP in dst
# 			dst = socket.inet_ntoa(ip.dst)

# 			# Print the source and destination IP
# 			print('Source: ' +src+ ' Destination: '  +dst)

# 		except:
# 			pass

# def main():
# 	# Open pcap file for reading
# 	f = open(path, 'rb')
# 	#pass the file argument to the pcap.Reader function
# 	pcap = dpkt.pcap.Reader(f)
# 	printPcap(pcap)

# if __name__ == '__main__':
# 	main()


#####
# def process_pcap(file_name):
#     print('Opening {}...'.format(file_name))

#     count = 0
#     for (pkt_data, pkt_metadata,) in RawPcapReader(file_name):
#         count += 1
#         print(pkt_data)
#         print(pkt_metadata)

#     print('{} contains {} packets'.format(file_name, count))

# if not os.path.isfile(file_path):
#         print('"{}" does not exist'.format(file_path), file=sys.stderr)
#         sys.exit(-1)
# process_pcap(file_path)

####
""" Tips from scapy Tutorial"""
file_path = "4_Tor_Website_Fingerprinting/train_data/appspot.com_2.pcap"

"""Unique IP Addresses"""
traffic = rdpcap(file_path)
#print(dir(traffic[0].payload.fields))
#print(traffic[0].payload.fields)
#print(traffic[0].payload.name)
options = []
for file in os.listdir(path):
     if file.endswith(".pcap"):
         new_path = os.path.join(path, file)
         traffic = rdpcap(file_path)
         #print(traffic[0].payload.name)
         
print(set(options))
#         port = traffic[TCP].sport
#         print(port)
#for file in os.listdir(path):
#    if file.endswith(".pcap"):
#        new_path = os.path.join(path, file)
# unique_ip_addresses = []
# src_ip_addresses = []
# dst_ip_addresses = []
# sent = 0
# rcv = 0

# for packet in traffic:
#     #print(packet)
    
#     src = packet[IP].src
#     dst = packet[IP].dst
#     unique_ip_addresses.append(src)
#     src_ip_addresses.append(src)
#     unique_ip_addresses.append(dst)
#     dst_ip_addresses.append(dst)
    #print(f'{src} : {dst}')
    # if src == "134.169.109.25": 
    #     sent += 1 
    # else: 
    #     rcv += 1

#print(set(unique_ip_addresses))
#print(f'{sent} : {rcv}')
#print(dst_ip_addresses)
"""Unique Port Numbers"""
# for file in os.listdir(path):
#     if file.endswith(".pcap"):
#         new_path = os.path.join(path, file)
#         unique_port_addresses = []
#         for packet in traffic[TCP]:
#             #print(packet)
#             src_port = packet[TCP].sport
#             dst_port = packet[TCP].dport
#             unique_port_addresses.append(src_port)
#             unique_port_addresses.append(dst_port)

#         print(set(unique_port_addresses))