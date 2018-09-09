import os
from shutil import copyfile
from random import sample
from math import floor
import numpy as np
import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
from keras import backend as K
import h5py

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

def split_train_test_dir(dir_of_data, train_dir, test_dir, train_percentage):
    """
    Split Create a training and testing directories for training keras model.
    Arguments:
    dir_of_data = Full path to the top directory of all the image folders.
    train_percentage = Float, representing the percentage of the images that 
    will be used for training.
    """   
        
    # Get a list of all the subdirectories
    image_dir = []
    print('Getting a list of the subdirectories...')
    for (f, d, r) in os.walk(dir_of_data):
        image_dir.append(d)

    image_dir = image_dir[0]

    # Create directories in the train and test folders
    print('Creating subdirectories...')
    for dir_ in image_dir:
        train_path_ = os.path.join(train_dir, dir_)

        if not os.path.isdir(train_path_):
            os.mkdir(train_path_)

        test_path_ = os.path.join(test_dir, dir_)

        if not os.path.isdir(test_path_):
            os.mkdir(test_path_)
    print('Created the subdirectories!')

    # Loop through each directory and create the split
    print('Looping through each subdirectory...\n')
    for img_dir in image_dir:

        full_path_img_dir = os.path.join(dir_of_data, img_dir)
        print("Splitting the images in: {0}\n".format(full_path_img_dir))

        for (_, _, files_list) in os.walk(full_path_img_dir):
            pass

        # Place training split into training folder
        'Get a percentage of the training images'
        num_images = len(files_list)
        num_training_images = floor(num_images*train_percentage)

        # Training Split
        training_images_files = sample(files_list, num_training_images)

        # Testing Split
        files_set = set(files_list)
        train_set = set(training_images_files)
        test_set = files_set - train_set
        testing_images_files = list(test_set)

        # Moves the files
        move_files(
            list_ = training_images_files,
            base_dir=full_path_img_dir,
            flag='train'
        )

        move_files(
            list_ = testing_images_files,
            base_dir=full_path_img_dir,
            flag='test'
        )

    # Get rid of extra test and train dirs
    os.rmdir(os.path.join(train_dir, 'train'))
    os.rmdir(os.path.join(train_dir, 'test'))
    os.rmdir(os.path.join(test_dir, 'train'))
    os.rmdir(os.path.join(test_dir, 'test'))

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

def multitask_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def build_X_y(data_directory):
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
            0,  <- Duck
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

    parent_class_list = ['goldeneye', 'grosbeak_bunting', 'towhee', 'grebe', 'scaup_duck']

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