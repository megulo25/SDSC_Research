# Here we need to produce the hierarchy as a json file for the website

import numpy as np
import os

cur_dir = os.getcwd()
a = cur_dir.split('/')
b = a[:-1]
c = '/'.join(b)

hierarchy_dict = np.load(os.path.join(c, 'hierarchy_dict.npy')).item()


class_dict_555 = np.load(os.path.join(c, 'data', 'class_dict_555.npy').item()

# Get leaf nodes

path_to_image_class_labels = os.path.join(c, 'nabirds_555', 'nabirds', 'image_class_labels.txt')

with open(path_to_image_class_labels, 'r') as txt_reader:
    class_labels_list = txt_reader.readlines()

set_ = set()
for i in class_labels_list:
    split_ = i.split()
    set_.add(split_[-1])


def iterate_class_hierarchy(node, dict_):
    if len(node) > 1:
        dict_['name']=node[0]
        branch = dict_.setdefault('children')
        iterate_class_hierarchy(node[1:], branch)
    else:
        dict_['name']='class_name'

dict_ = {}

# Construct hierarchy json
for j in set_:

    # Set will be unordered
    class_hierarchy_set = a[j]

    # Convert to list and sort
    class_hierarchy_list = list(class_hierarchy_set)
    class_hierarchy_list.sort()

    # Go through each leaf node and construct the hierarchy
    iterate_class_hierarchy(node=class_hierarchy_list, dict_=dict_)