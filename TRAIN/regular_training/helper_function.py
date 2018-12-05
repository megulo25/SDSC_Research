from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import h5py
import os
import time

def check_data_folder():
    if not os.path.isdir('data'):
        os.mkdir('data')

def check_nabirds10_folder():
    if not os.path.isdir('data/nabirds10'):
        os.chdir('data')
        os.mkdir('nabirds10')
        os.chdir('nabirds10')
        os.system('wget https://www.dropbox.com/sh/g6aatnar4n5s63g/AABBixZUh5SiPvFS7eVVVxlHa')
        os.system('mv -v AABBixZUh5SiPvFS7eVVVxlHa nabirds10.zip')
        os.system('unzip nabirds10.zip')
        os.remove('nabirds10.zip')
        os.removedirs('Western Grebe')
        os.removedirs('SpottedTowheee')
        os.removedirs('CommonGoldeneye')
        os.chdir('../..')

def split_data(test_split):

    # Get directory structure
    directories = os.listdir('data/nabirds10')
    
    # Add train/validation/test sets
    if not os.path.isdir('data/nabirds10/train'):
        os.mkdir('data/nabirds10/train')

        for i in directories:
            os.mkdir(os.path.join('data/nabirds10/train', i))

    if not os.path.isdir('data/nabirds10/validation'):
        os.mkdir('data/nabirds10/validation')

        for j in directories:
            os.mkdir(os.path.join('data/nabirds10/validation', j))

    if not os.path.isdir('data/nabirds10/test'):
        os.mkdir('data/nabirds10/test')

        for k in directories:
            os.mkdir(os.path.join('data/nabirds10/test', k))

    # Split data
    for dir_ in directories:
        img_files = os.listdir(os.path.join('data/nabirds10', dir_))
        
        train_list, other_list = train_test_split(img_files, test_size=test_split)
        val_list, test_list = train_test_split(other_list, test_size=.5)
        del other_list

        # Move to train folder
        for tr in train_list:
            os.rename(os.path.join('data/nabirds10', dir_, tr), os.path.join('data/nabirds10', 'train', dir_, tr))
        
        # Move to validation folder
        for vl in val_list:
            os.rename(os.path.join('data/nabirds10', dir_, vl), os.path.join('data/nabirds10', 'validation', dir_, vl))

        # Move to test folder
        for te in test_list:
            os.rename(os.path.join('data/nabirds10', dir_, te), os.path.join('data/nabirds10', 'test', dir_, te))


def build_X_y_10(data_directory):
    """
    Here we will return an X and y for our multitask learning data.
    The data will be forced to a size of (224,224,3) for our model.

    X: (n, 224, 224, 3)
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
    X = np.zeros((1, 224, 224, 3)).astype(np.uint8)
    y = np.zeros((1,15)).astype(np.uint8)

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

    full_path_list = os.listdir(data_directory)

    n = len(full_path_list)
    c = 0
    # Loop through each subdirectory and build X and y
    for dir_ in full_path_list:

        list_of_images = os.listdir(os.path.join(data_directory, dir_))

        # Loop through each image
        for img_name in list_of_images:
            img_full_path = os.path.join(data_directory, dir_, img_name)
            img_old = mpimg.imread(img_full_path)

            # Resize the image
            img = cv2.resize(img_old, (224, 224))
            img = np.reshape(img, (1,224,224,3))

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
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)
    h5f = h5py.File('data_10.h5', 'w')
    h5f.create_dataset('X', data=X)
    h5f.create_dataset('y', data=y)
    h5f.close()