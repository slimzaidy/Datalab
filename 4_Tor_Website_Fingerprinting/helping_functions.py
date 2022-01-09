import dpkt
import socket
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import os

def read_pcap_file(file, path):
    """ Read the pcap file and return the sizes of the packets. """

    # This is the pcap file we'll be reading at this point.
    file = os.path.join(path, file)
    # Read the file.
    fp = open(file, 'rb')

    # Create the pcap object
    pcap = dpkt.pcap.Reader(fp)

    # This is the array that will contain all the packet sizes.
    sizes = [0] * 40
    i = 0

    # Hold the addresses of the outgoing agent.
    outgoing_addr = None

    outgoing_packets = 0
    incoming_packets = 0
    total_number_of_packets = 0

    # This will contain the total size of the incoming packets.
    incoming_size = 0

    # Loop through all the packets and save the sizes.
    for ts, buf in pcap:
        packet_size = len(buf)
        is_outgoing = True

        # Parse the Ethernet packet.
        eth = dpkt.ethernet.Ethernet(buf)

        # Parse the IP packet.
        ip = eth.data

        # Get the source addresses.
        src = inet_to_str(ip.src)

        if total_number_of_packets == 0:
            # Get the address of the outgoing agents. The target user is the
            # outgoing agent, and the incoming packets are the server/website.
            outgoing_addr = src
            outgoing_packets += 1

        elif src == outgoing_addr:
            # Increment the outgoing packets.
            outgoing_packets += 1

        else:
            # Increment the incoming packets.
            incoming_packets += 1

            # Increment the size of the incoming packets.
            incoming_size += packet_size

            # This is an incoming packet.
            is_outgoing = False

        if i < 40:
            # Add the size to the array.
            sizes[i] = packet_size if is_outgoing else -packet_size

            # Increment the index.
            i += 1

        # Increment the total amount of packets.
        total_number_of_packets += 1

    # Get the ratio.
    ratio = float(incoming_packets) / (outgoing_packets if outgoing_packets != 0 else 1)

    # Print some details.
    print(f'OUT: {outgoing_packets},' +
            f'IN: {incoming_packets},' +
            f'TOTAL: {total_number_of_packets},' +
            f'SIZE: {incoming_size},' +
            f'RATIO: {ratio}')

    # Reverse the array to append the other information.
    sizes.reverse()

    # Add the ratio of incoming to outgoing packets.
    sizes.append(ratio)

    # Add the number of incoming packets.
    sizes.append(incoming_packets)

    # Add the number of outgoing packets.
    sizes.append(outgoing_packets)

    # Add the number of total packets.
    sizes.append(total_number_of_packets)

    # Add the total size of the incoming packets.
    sizes.append(incoming_size)

    # Reverse the array again so that the sizes are in order.
    sizes.reverse()

    # Finally return the sizes.
    return sizes