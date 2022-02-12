Network-based attack detection

In this task, a learning-based intrusion detection system for the FTP protocol is to be developed. To keep things simple, each FTP session corresponds exactly to a TCP connection. We left out the typical data channel of FTP, which actually carries most of the data. So you only see the client's control commands and the server's responses.

The system should take a pcap file as input and then output predictions for each TCP connection in the following format:

    192.168.178.38:43604->192.168.178.30:21;0
    192.168.178.74:51869 -> 192.168.178.30:21;1
    192.168.178.99:55989 -> 192.168.178.30:21;0
    192.168.178.88:47611->192.168.178.30:21;1
    ...
The first field describes a TCP connection and the second field is your prediction, with 0 representing a harmless connection and 1 representing an attack. In the example, the connection from 192.168.178.38:43604 to 192.168.178.30:21 is considered normal.

Attention: Unfortunately there are 4 connections that are not clear. The same combination of IP and TCP port is used here as with another connection. These compounds can be classified as benign. 