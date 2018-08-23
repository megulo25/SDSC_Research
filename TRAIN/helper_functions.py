import os
from shutil import copyfile
from random import sample
from math import floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
import itertools
import cv2
from keras import backend as K

def split_train_test_dir(dir_of_data, train_dir, test_dir,train_percentage):
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
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def y_one_hot_enc(dict_, class_):
    'return one hot encoding'
    arr = np.zeros((1011))
    list_ = list(dict_[class_])
    arr[list_] = 1
    return arr

def train_test_split_multi_output(full_path_to_data):

    # Ask to load in existing X and y
    message = input('Where do you want to continue from? ')
    message = int(message)
    try:
        X = np.load('X.npy')
        y = np.load('y.npy')
        exist_ = True
    except:
        exist_ = False
        print('There is no X nor y')
    
    # Import dict
    dict_ = np.load('hierarchy_dict.npy').item()

    # Get a list of all subdirectories
    image_dir = []

    for (f, d, r) in os.walk(full_path_to_data):
        image_dir.append(d)

    image_dir = image_dir[0]

    # Loop through each image
    if not exist_:
        X = np.zeros((1,227,227,3))
        y = []
    count = 0
    c = 0
    num_dirs = len(image_dir)
    for img_dir in image_dir:
        print(c)
        c+=1
        if count >= message:
            print('Directory: {0}/{1}'.format(count, num_dirs))

            # Get each image
            full_path = os.path.join(full_path_to_data, img_dir)
            for (_, _, img_files) in os.walk(full_path):
                pass

            # Resize all the images
            for img_file in img_files:
                try:
                    img = mpimg.imread(os.path.join(full_path,img_file))
                    img_resized = cv2.resize(img, (227,227))
                    del img
                    img_reshaped = img_resized.reshape((1,227,227,3))
                    del img_resized

                    # Concatenate X
                    X = np.concatenate((X, img_reshaped))
                    del img_reshaped

                    # Concatenate y
                    class_ = int(img_dir)
                    y_output = y_one_hot_enc(dict_, class_)
                    y.append(y_output)

                    if (c == 0) and (not exist_):
                        X = X[1:]
                        c=1

                except:
                    print('Error with {0}'.format(img_file))

        count+=1

        np.save('X.npy', X)
        np.save('y.npy', y)

    np.save('X.npy', X)
    np.save('y.npy', y)