# This script will contain functions for building datasets 
# for different types of learning tasks
#------------------------------------------------------------------------------------#
# Import Modules
import functools
import os
import time

import h5py
import matplotlib.image as mpimg
import numpy as np
from cv2 import resize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from helper_functions import setup_data_folder_structure
#------------------------------------------------------------------------------------#
# Housekeeping

# Set up datapaths
setup_data_folder_structure()

# Data Paths
BASE_DIR = os.getcwd()
DATASET_10 = os.path.join(BASE_DIR, 'datasets', 'dataset_10')
DATASET_555 = os.path.join(BASE_DIR, 'datasets', 'dataset_555', 'images')
CLASSES_TXT_PATH = os.path.join(BASE_DIR, 'datasets', 'dataset_555', 'classes.txt')

# Output Paths
STL_10 = os.path.join(BASE_DIR, 'datasets', 'STL_10')
STL_555 = os.path.join(BASE_DIR, 'datasets', 'STL_555')
MTL_10 = os.path.join(BASE_DIR, 'datasets', 'MTL_10')
MTL_555 = os.path.join(BASE_DIR, 'datasets', 'MTL_555')

# Check if they exist, if not create them
if not os.path.isdir(STL_10):
    os.mkdir(STL_10)

if not os.path.isdir(STL_555):
    os.mkdir(STL_555)

if not os.path.isdir(MTL_10):
    os.mkdir(MTL_10)

if not os.path.isdir(MTL_555):
    os.mkdir(MTL_555)

# Initialization
IMAGE_SIZE = (299, 299)
IMAGE_SHAPE = (1, 299, 299, 3)
STL_10_OUTPUT_VECTOR_SIZE = (1, 10)
STL_555_OUTPUT_VECTOR_SIZE = (1, 555)

# A little simple decorator
def printDecorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('------------------------------')
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('Runtime: {0} secs'.format(run_time))
    return wrapper

@printDecorator
def printProgress(class_index, class_len, image_index, image_len):
    print('Class: {0}/{1}'.format(class_index, class_len))
    print('Image: {0}/{1}'.format(image_index, image_len))

#------------------------------------------------------------------------------------#
# One-Task Learning, 10 classes (Passed: True)
def build_STL_10():
    """
    Builds .h5 file containing 3 keys for a 10 class dataset:
    X - Training data
    y - Output label
    y_label - Output class name

    'y' is not one-hot encoding because autokeras does not allow multi output vector.
    """
    
    # Initialize variables
    X = np.zeros(IMAGE_SHAPE)
    y = []

    # Get all labels from subdirectory names
    labels = os.listdir(DATASET_10)
    y_labels = []

    # Loop through each subdirectory
    for label_index, subdir in enumerate(labels):
        
        # Get all image names
        IMAGES_DIR = os.path.join(DATASET_10, subdir)
        images_list = os.listdir(IMAGES_DIR)

        # Loop through each image
        for image_index, image in enumerate(images_list):

            # Read in image
            img = mpimg.imread(os.path.join(IMAGES_DIR, image))

            # Resize
            img_resized = resize(img, IMAGE_SIZE)
            img_reshaped = np.reshape(img_resized, IMAGE_SHAPE)

            # Label
            y_labels.append(subdir)
            y.append(label_index)

            # Build variables
            X = np.concatenate([X, img_reshaped])

            # Print Progress
            if image_index % 20 == 0:
                printProgress(label_index, len(labels), image_index, len(images_list))
    
    # Remove initial zeros
    X = X[1:]
    y_labels = np.array(y_labels, dtype='S')

    # Save to disk
    hfile = h5py.File(os.path.join(STL_10, 'data_STL_10.h5'), 'w')
    hfile.create_dataset('X', data=X)
    hfile.create_dataset('y', data=y)
    hfile.create_dataset('y_labels', data=y_labels)
    hfile.close()
#------------------------------------------------------------------------------------#
# One-Task Learning, 555+ classes (Passed: True)

def getClassDict():
    """
    This creates a dictionary containing:
    key - Subdirectory name (e.g. 0295)
    value - Class name (e.g. Western Grebe)
    """
    class_dict = {}
    # Read in classes.txt
    with open('{0}'.format(CLASSES_TXT_PATH), 'r') as file_reader:
        class_txt = file_reader.readlines()
    for _, value in enumerate(class_txt):
        # Split
        value_split = value.split()
        number = value_split[0]
        # Rejoin
        value_rejoined = ' '.join(value_split[1:])
        # Create temp dict
        dict_ = {
            number: value_rejoined
        }
        # Append to dict
        class_dict = {**class_dict, **dict_}
    return class_dict

def build_STL_555():
    """
    Build a .h5 file containg 3 keys for a 555 class dataset:
    X - Training data
    y - Output label
    y_label - Output class name
    """

    # Initialize Variables
    X = np.zeros(IMAGE_SHAPE)
    y = np.zeros(STL_555_OUTPUT_VECTOR_SIZE)

    # Bring in class_dict
    class_dict = getClassDict()

    # Get all labels from subdirectory names
    labels = os.listdir(DATASET_555)
    y_labels = []

    # Initialize Label encoder
    label_encoder = LabelEncoder()
    subdir_encodings = label_encoder.fit_transform(labels)

    # Loop through each subdirectory
    for label_index, subdir in enumerate(labels):
        
        # Get all images names
        IMAGES_DIR = os.path.join(DATASET_555, subdir)
        images_list = os.listdir(IMAGES_DIR)

        # Loop through each image
        for image_index, image in enumerate(images_list):

            # Read in image
            img = mpimg.imread(os.path.join(IMAGES_DIR, image))

            # Resize
            img_resized = resize(img, IMAGE_SIZE)
            img_reshaped = np.reshape(img_resized, IMAGE_SHAPE)

            # Label
            y_labels.append(class_dict[str(int(subdir))])
            y_output_vector = np.zeros(STL_555_OUTPUT_VECTOR_SIZE)

            # Encode
            idx = subdir_encodings[label_index]
            y_output_vector[0, idx] = 1

            # Build variables
            X = np.concatenate([X, img_reshaped])
            y = np.concatenate([y, y_output_vector])

            # Print Progress
            if image_index % 20 == 0:
                printProgress(label_index, len(labels), image_index, len(images_list))

    # Remove initial zeros
    X = X[1:]
    y = y[1:]
    y_labels = np.array(y_labels, dtype='S')

    # Save to disk
    hfile = h5py.File(os.path.join(STL_555, 'data_STL_555.h5'), 'w')
    hfile.create_dataset('X', data=X)
    hfile.create_dataset('y', data=y)
    hfile.create_dataset('y_labels', data=y_labels)
    hfile.close()
#------------------------------------------------------------------------------------#
# Multi-Task Learning, 10 classes (Passed: )
#------------------------------------------------------------------------------------#
# Multi-Task Learning, 555+ classes (Passed: )
#------------------------------------------------------------------------------------#

if __name__ == "__main__":
    build_STL_10()
    # build_STL_555()