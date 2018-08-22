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