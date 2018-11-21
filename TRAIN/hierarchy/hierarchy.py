import os

cur_dir = os.getcwd()
a = cur_dir.split('/')
b = a[:-1]
c = '/'.join(b)

with open(os.path.join(c, 'data', 'nabirds_555', 'nabirds', 'hierarchy.txt'), 'r') as file_reader:
    list_ = file_reader.readlines()

new_list = []
for item in list_:
    split = item.split()
    one = int(split[0])
    two = int(split[1])
    new_list.append((one, two))

del list_
dict_ = {}
c=0
for row in new_list:
    # print(c)
    # c+=1
    key = row[0]
    val = row[1]

    set_ = set()
    set_.add(val)

    dict_[key] = set_

    while 0 not in set_:
        val = dict_[val]
        set_ = set_.union(val)
        dict_[key] = set_

import numpy as np
np.save('hierarchy_dict.npy', dict_)