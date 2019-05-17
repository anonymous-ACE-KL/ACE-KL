import sys
import time
import sets
from collections import OrderedDict
import numpy as np
import progressbar


def main():

    bar = progressbar.ProgressBar()
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        lines = f.readlines()

    ipset = sets.Set()
    for line in bar(lines):
        new_line = line.replace('\n', '').split(',')
        # print new_line
        srcIP = new_line[2]
        dstIP = new_line[3]
        # print srcIP, dstIP
        if srcIP not in ipset:
            ipset.add(srcIP)
        if dstIP not in ipset:
            ipset.add(dstIP)

    print "different IPs: {}".format(len(ipset))

    bar = progressbar.ProgressBar()
    for user_ip in bar(ipset):
        flow_list = []
        for line in lines:
            new_line = line.replace('\n', '').split(',')
            srcIP = new_line[2]
            dstIP = new_line[3]
            if (srcIP == user_ip) or (dstIP == user_ip):
                flow_list.append(new_line)
        with open("userdata/"+user_ip + ".csv", "w") as h:
            for record in flow_list:
                h.write("\t".join(record) + "\n")



if __name__ == "__main__":
    main()
