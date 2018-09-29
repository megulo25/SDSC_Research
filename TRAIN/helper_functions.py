import os
from shutil import copyfile
from random import sample
from math import floor
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
from keras import backend as K
import h5py
import pandas as pd
import shutil
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(message):
    if not os.path.isdir('data'):
        os.mkdir('data')
    os.chdir('data')
    if message == 0:
        message = 'nabirds_10'
        class_count = 10
        if not os.path.isdir('nabirds_10'):
            os.mkdir('nabirds_10')

        if len(os.listdir('nabirds_10')) == 0:
            os.chdir('nabirds_10')
            os.system('wget https://www.dropbox.com/sh/g6aatnar4n5s63g/AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
            os.system('unzip AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
            os.remove('AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
            shutil.rmtree('CommonGoldeneye')
            shutil.rmtree('SpottedTowheee')
            shutil.rmtree('Western Grebe')
            full_path_to_data = os.path.join(os.getcwd())
            training_dir = os.path.join(full_path_to_data, 'train')
            validation_dir = os.path.join(full_path_to_data, 'test')
            os.chdir('../..')
        else:
            full_path_to_data = os.path.join(os.getcwd(), message)
            training_dir = os.path.join(full_path_to_data, 'train')
            validation_dir = os.path.join(full_path_to_data, 'test')
            os.chdir('..')
    
    elif message == 1:
        message = 'nabirds_555'
        class_count = 555
        if not os.path.isdir('nabirds_555'):
            os.mkdir('nabirds_555')

        if len(os.listdir('nabirds_555')) == 0:
            os.chdir('nabirds_555')
            os.system('wget https://www.dropbox.com/s/nf78cbxq6bxpcfc/nabirds.tar.gz')
            os.system('tar xvzf nabirds.tar.gz')
            os.remove('nabirds.tar.gz')
            full_path_to_data = os.path.join(os.getcwd(), 'nabirds', 'images')
            training_dir = os.path.join(full_path_to_data, 'train')
            validation_dir = os.path.join(full_path_to_data, 'test')
            os.chdir('../..')
        else:
            full_path_to_data = os.path.join(os.getcwd(), 'nabirds_555','nabirds', 'images')
            training_dir = os.path.join(full_path_to_data, 'train')
            validation_dir = os.path.join(full_path_to_data, 'test')
            os.chdir('..')
    else:
        raise NameError('You need to enter either 0 or 1!')
    
    return training_dir, validation_dir

def split_train_test_dir():
    """
    Split Create a training and testing directories.
    Arguments:
    dir_of_data = Full path to the data directory.
    """   
    dir_of_data = os.path.join(os.getcwd(), 'data')
    path_to_train_test_split_file = os.path.join(dir_of_data, 'nabirds_555', 'nabirds', 'train_test_split.txt')

    # Bring in text file that has the split defined
    with open(path_to_train_test_split_file, 'r') as file:
        train_test_split_list = file.readlines()

    # Convert list to dictionary (key=filename, value=1/0)
    dict_ = {}
    for i in train_test_split_list:
        i = i.split()
        key_ = i[0].replace('-','')
        val_ = int(i[1])
        dict_[key_] = int(val_)

    # Get list of subdirectories for one-hot encoding
    for (_, subdirs, _) in os.walk(os.path.join(dir_of_data, 'nabirds_555', 'nabirds', 'images')):
        True
        break
    
    subdirs = np.array(subdirs)
    subdirs = subdirs.astype('int')

    label_encoder = LabelEncoder()
    subdir_encodings = label_encoder.fit_transform(subdirs)

    label_dict = {}
    for j_idx, j_val in enumerate(subdirs):
        label_dict[j_val] = subdir_encodings[j_idx]

    # Build X_train, y_train, X_test, y_test   
    class_count = 555

    X_train = np.zeros((1, 224, 224, 3))
    X_test = np.zeros((1, 224, 224, 3))
    y_train = np.zeros((1, class_count))
    y_test = np.zeros((1, class_count))

    c=0
    tr=0
    te=0
    for (full_path_sub_dirs, _, img_file_names) in os.walk(os.path.join(dir_of_data, 'nabirds_555', 'nabirds', 'images')):
        if len(img_file_names) > 10:
            for img_name in img_file_names:

                try:      
                    full_img_path = os.path.join(full_path_sub_dirs, img_name)

                    img = mpimg.imread(full_img_path)
                    img = cv2.resize(img, (224, 224))
                    img = np.reshape(img, (1, 224, 224, 3))

                    y_ex = np.zeros((1, class_count))
                    true_class = int(full_path_sub_dirs.split('/')[-1])
                    label_index = label_dict[true_class]
                    y_ex[0, label_index] = 1

                    # Place into either train or test set
                    if dict_[img_name[:-4]] == 1:
                        X_train = np.concatenate([X_train, img])
                        y_train = np.concatenate([y_train, y_ex])
                        tr+=1
                    else:
                        X_test = np.concatenate([X_test, img])
                        y_test = np.concatenate([y_test, y_ex])
                        te+=1
                except:
                    print('Error with image: {0}'.format(os.path.join(full_path_sub_dirs, img_name)))
            c+=1
            print('Completed: {0}/{1} directories'.format(c, class_count))
    
    print('Num. of training ex: {0}'.format(tr))
    print('Num. of testing ex: {0}'.format(te))
    X_train = X_train[1:]
    X_test = X_test[1:]
    y_train = y_train[1:]
    y_test = y_test[1:]

    h = h5py.File('data_555.h5', 'w')
    h.create_dataset('X_train', data=X_train)
    h.create_dataset('X_test', data=X_test)
    h.create_dataset('y_train', data=y_train)
    h.create_dataset('y_test', data=y_test)
    h.close()

def move_files(list_, base_dir, flag):
    """
    Moves the files to the proper directory
    """
    for filename in list_:

        base_dir_split = base_dir.split('/')
        part_1_dir = '/'.join(base_dir_split[:-1])
        part_2_dir = base_dir_split[-1]

        input_full_path = os.path.join(base_dir, filename)
        output_full_path = os.path.join(part_1_dir, flag, part_2_dir, filename)

        copyfile(input_full_path, output_full_path)

def build_X_y_10(data_directory):
    """
    Here we will return an X and y for our multitask learning data.
    The data will be forced to a size of (299,299,3) for our model.

    X: (n, 299, 299, 3)
    y: (n, 15) ==> 5 for each parent class, 10 for each child class

    Ex:
    y = [   
            0,  <- Goldeneye
            0,  <- Grosbeak_Bunting
            0,  <- Towhee
            0,  <- Grebe
            0,  <- Scaup_Duck
            0,  <- Barrows_Goldeneye
            0,  <- Blue_Grosbeak
            0,  <- Clarks_Grebe
            0,  <- Common_Goldeneye
            0,  <- Eastern_Towhee
            0,  <- Indigo_Bunting
            0,  <- Lesser_Scaup
            0,  <- Ring_Necked_Duck
            0,  <- Spotted_Towhee 
            0   <- Western_Grebe
    ]
    """
    X = np.zeros((1, 299, 299, 3))
    y = np.zeros((1,15))

    child_class_dict = {
        'barrows_goldeneye': 5,
        'blue_grosbeak': 6,
        'clarks_grebe': 7,
        'common_goldeneye': 8,
        'eastern_towhee': 9,
        'indigo_bunting': 10,
        'lesser_scaup': 11,
        'ring_necked_duck': 12,
        'spotted_towhee': 13,
        'western_grebe': 14
    }

    full_path_list = []

    for (full, _, _) in os.walk(data_directory):
        full_path_list.append(full)
    
    # Get rid of main dir from full_path_list
    full_path_list = full_path_list[1:]

    n = len(full_path_list)
    c = 0
    # Loop through each subdirectory and build X and y
    for dir_ in full_path_list:
        for (_, _, list_of_images) in os.walk(dir_):
            pass
        # Loop through each image
        for img_name in list_of_images:
            img_full_path = os.path.join(dir_, img_name)
            img_old = mpimg.imread(img_full_path)
            
            # Resize the image
            img = cv2.resize(img_old, (299, 299))
            img = np.reshape(img, (1,299,299,3))

            # Add to X
            X = np.concatenate([X, img])

            # Add to y
            list_ = dir_.split('/')
            relative_name = list_[-1].lower()
            relative_name_list = relative_name.split('_')

            for i in relative_name_list:
                if i in 'goldeneye':
                    p_idx = 0
                    break
                elif i in 'grosbeak':
                    p_idx = 1
                    break
                elif i in 'bunting':
                    p_idx = 1
                    break
                elif i in 'towhee':
                    p_idx = 2
                    break
                elif i in 'grebe':
                    p_idx = 3
                    break
                elif i in 'scaup':
                    p_idx = 4
                    break
                elif i in 'duck':
                    p_idx = 4
                    break

            y_i = np.zeros((1,15))
            y_i[0, p_idx] = 1

            c_idx = child_class_dict[relative_name]
            y_i[0, c_idx] = 1

            y = np.concatenate([y, y_i])
        
        c+=1
        print('Completed: {0}/{1}'.format(c, n))

    X = X[1:]
    y = y[1:]
    h5f = h5py.File('data_10.h5', 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('y', data=y)
    h5f.close()

def build_X_y_555(data_directory):
    """
    Building for 555 classes. Not for MTL.
    """

    # Bring in hierarchy for multi-task learning
    hierarchy_dict = np.load(os.path.join(os.getcwd(), 'data', 'hierarchy_dict.npy')).item()

    X = np.zeros((1, 299, 299, 3))
    y = np.zeros((1, 1011))

    full_path_list = []

    # One hot encode each subdirectory
    for (_, dir_names, _) in os.walk(data_directory):
        1
        break
    dir_names.sort()


    for (full, _, _) in os.walk(data_directory):
        full_path_list.append(full)

    # Get rid of main dir from full_path_list
    full_path_list = full_path_list[1:]

    n = 49117
    c = 0
    cc=0
    # Loop through each subdirectory and build X and y
    for dir_ in full_path_list:
        for (_, _, list_of_images) in os.walk(dir_):
            pass

        # Loop through each image
        for img_name in list_of_images:
            try:

                img_full_path = os.path.join(dir_, img_name)
                img_old = mpimg.imread(img_full_path)

                # Resize and reshape img
                img = resize_and_reshape_image(img_old)

                # Add to X
                X = np.concatenate([X, img])

                # Add to y
                leaf_node = int(dir_.split('/')[-1])
                full_hierarchy = list(hierarchy_dict[leaf_node])

                temp_y = np.zeros((1, 1011))
                temp_y[0, leaf_node]=1
                temp_y[0, full_hierarchy]=1
                y = np.concatenate([y, temp_y])

                c+=1

                if c % 3000 == 0:
                    print('Completed: {0}/{1}'.format(c, n))
                    print('Saving...')
                    f = h5py.File('data_555_MTL_{0}.h5'.format(cc), 'w')
                    f.create_dataset('X', data=X)
                    f.create_dataset('y', data=y)
                    f.close()
                    print('Done. Working on the next batch.')
                    cc+=1
                    del X
                    del y

                    X = np.zeros((1, 299, 299, 3))
                    y = np.zeros((1, 1011))
            except:
                print('This image failed: {0}'.format(img_name))

    # Save X and y
    X = X[1:]
    y = y[1:]

    h5f = h5py.File('data_555_MTL_{0}.h5'.format(cc), 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('y', data=y)
    h5f.close()

def random_images(n, dir_):
    full_path_list = []
    list_ = []

    for (f, _, _) in os.walk(dir_):
        full_path_list.append(f)
        

    full_path_list = full_path_list[1:]

    for item in full_path_list:
        for (_,_,li) in os.walk(item):
            pass
        for l in li:
            list_.append(os.path.join(item, l))

    return sample(list_, n)

def classes_to_dict(path_to_file):
    """
    The purpose of this function is to take the classes.txt file 
    for the 500+ class dataset and turn it into a dictionary for 
    the one hot encoding.
    """

    with open(path_to_file, 'r') as file_reader:
        list_ = file_reader.readlines()

    new_list = []
    dict_ = {}
    for item in list_:
        split = item.split()
        index = int(split[0])
        label = '_'.join(split[1:])
        label = label.replace(',', '')
        dict_[index] = label

    # Save dict
    np.save('class_dict_555.npy', dict_)

def create_final_hidden_layer():
    # Hierarchy file
    a = os.getcwd()
    b = a.split('/')
    b = b[:-2]
    c = '/'.join(b)
    hierarchy_dict = np.load(os.path.join(c, 'data', 'hierarchy_dict.npy')).item()
    # Get a list of all the classes 
    path = os.path.join(c, 'data', 'nabirds_555', 'nabirds', 'images')
    list_ = []
    for (_, l, _) in os.walk(path):
        list_.append(l)
    list_ = list_[0]
    # Create array
    final_hidden_layer = np.zeros((len(list_), 1011))
    # Get the heirarchy for each class and place into array
    c = 0
    for i, j in enumerate(list_):
        # Convert to integer
        j = int(j)
        # Get hierarchy 
        y = list(hierarchy_dict[j])
        # Index array
        final_hidden_layer_old = final_hidden_layer[i]
        final_hidden_layer[i][y] = 1
    return final_hidden_layer

def initialze_final_hidden_layer(weights, array_of_all_hierarchies_in_training_set, flag='init_a'):
    # Grab the weights from the last hidden layer
    last_hidden_layer_weights = weights[-2]
    n = len(array_of_all_hierarchies_in_training_set)
    # Replace the first n rows with possible hierarchies
    last_hidden_layer_weights[:n] = array_of_all_hierarchies_in_training_set
    # Initialize the rest of the rows with one of the following
    if (flag == 'init_a'):
        # Initialize the rest with zeros
        na = last_hidden_layer_weights[n:].shape
        last_hidden_layer_weights[n:] = np.zeros(na)
    elif (flag == 'init_b'):
        # Initialize the rest with random values.
        nb = last_hidden_layer_weights[n:].shape
        last_hidden_layer_weights[n:] = np.random.rand(nb)
    # Replace last hidden layer in weights list
    weights[-2] = last_hidden_layer_weights
    return weights