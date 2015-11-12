#!/usr/bin/python

import jaws
import os

DIR = "/home/hlpark/workspace/scripts"
line = open(DIR+"/log").read().splitlines()

print("\n\033[0;37m%s\n" % ("***** START *****"));

item = ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10", "item11", "citem12", "item13","item14", "item15", "item16","item17"]
out_item = '\t'.join(map(str, item))
print("\033[1;31m%s\033[0;38m" % (out_item));

idx = 1
result = []
for l in line:
    data = jaws.findall(l, r'[0-9]*\.[0-9]+.')
    if data:
        result.append(data)
        if (idx % len(item) == 0):
            output = jaws.printCSV(result, '\t')
            print output
            result = []
        idx += 1
print("\n\033[0;37m%s\n" % ("***** END *****"));
