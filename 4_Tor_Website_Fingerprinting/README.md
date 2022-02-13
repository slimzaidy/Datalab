Tor site fingerprinting

Closed World

In the closed world scenario, the user can only visit certain websites. The attacker is aware of these websites and uses certain characteristics in the encrypted network traffic to try to reconstruct which pages the user has visited.
In this task you should first learn a classifier on labeled training data that can distinguish the given 30 pages from each other as well as possible. This classifier is then applied to unlabeled test data to predict web page names.
The format for the predictions is as follows:

     1.pcap;www.sectubs.de
     2.pcap;www.greatagain.gov
     3.pcap;www.tu-braunschweig.de
     ...
The first field is the file name and the second field is your prediction. In the example, the recorded network traffic in 1.pcap was predicted as a visit to www.sectubs.de.