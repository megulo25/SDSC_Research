# Here we need to produce the hierarchy as a json file for the website
import numpy as np
import os
import time

cur_dir = os.getcwd()
a = cur_dir.split('/')
b = a[:-1]
c = '/'.join(b)

# Get files
with open(os.path.join(c, 'data', 'nabirds_555', 'nabirds', 'hierarchy.txt'), 'r') as f:
    l = f.readlines()

class_dict_555 = np.load(os.path.join(c, 'data', 'class_dict_555.npy')).item()

list_ = []
for i in l:
    i = i.split()
    i[0] = int(i[0])
    i[1] = int(i[1])
    list_.append(i)

array = np.array(list_)

idx = np.argsort(array, axis=0)
idx = idx[:,1]

arr = array[idx]

dict_ = {}
old_j = None
f=0
list_dict = []
for j in arr:
    node_name = j[1]
    if old_j == node_name:
        # Adding to current parent node
        dict_['children'].append({
            'name':j[0],
            'children':[]
        })
    elif (old_j != node_name) and (f==0):
        # Create a new branch
        dict_['name'] = node_name
        dict_['children'] = [{
            'name': j[0],
            'children': []
        }]
        f=1
        old_j = node_name
    else:
        np.save('H_{0}.npy'.format(node_name-1), dict_)
        dict_ = {}
        f=0