import pyshark

path = "C:/Users/askar/OneDrive/Dokumente/Meine Dateien/Uni/IST/Datalab/Datalab/4_Tor_Website_Fingerprinting/train_data/appspot.com_2.pcap"

cap = pyshark.FileCapture(path)

print(cap)

labels = []

string = "appspot.com_2.pcap"

label = string.split("_")[0]
print(label)