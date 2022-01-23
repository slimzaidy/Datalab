
rez = []

with open("5_Netzbasierte_Angriffserkennung\output.txt") as f:
    complete_string = ''

    for x in f:
        x = x.strip("=")
        complete_string += x
    complete_string = " ".join(complete_string.split())[136:].strip()

rez.append(complete_string)

print(rez)
#print(complete_string)

#for /L %i in (0,1,9999) do tshark -r nids-test.pcap -q -z follow,tcp,ascii,%i > strings_test\%i_output.txt
