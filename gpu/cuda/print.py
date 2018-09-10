#!/usr/bin/python

import jaws
import os

DIR = "./"
line = open(DIR+"/log").read().splitlines()

print("\n\033[0;37m%s\n" % ("***** START *****"));

X = [128, 256, 512, 1024, 2048]
Y = ["global", "local", "texture"]

outY = '\t'.join(map(str, Y))
print("\033[1;34m#item\t\033[1;31m%s" % (outY));


idx = 0
idy = 0
result = []
for l in line:
    data = jaws.findall(l, r'[0-9]*\.[0-9]+.')
    if data:
        result.append(data)
        idy += 1
        if (idy % len(Y) == 0):
            output = jaws.printCSV(result, '\t')
            print("\033[1;31m%s\t\033[0;38m%s" % (X[idx], output))
            idx += 1
            if (idx % len(X) == 0):
                idx = 0
                print("\033[0;37m-----")
            result = []
print("\n\033[0;37m%s\n" % ("***** END *****"));
