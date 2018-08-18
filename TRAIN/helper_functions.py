import os
from shutil import copyfile
from random import sample
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2


def split_train_test_dir(dir_of_data, train_percentage):
    """
    Split Create a training and testing directories for training keras model.

    Arguments:
    dir_of_data = Full path to the top directory of all the image folders.
    train_percentage = Float, representing the percentage of the images that 
    will be used for training.
    """

    # Create Train and Test folders.
    print('Checking if folders already exist...')
    train_dir = os.path.join(os.getcwd(), 'data', 'nabirds', 'images', 'train')
    if not os.path.isdir(train_dir):
        print("Training folder does not exist. Adding it now. You can find it at {0}".format(train_dir))
        os.mkdir(train_dir)
    else:
        message = """
        Train folder already exists. \nWould you like to remove it and create a new one? (y/n)
        """

        train_resp = input(message)
        train_resp = train_resp.lower()

        if train_resp == 'y':
            os.remove(train_dir)
            os.mkdir(train_dir)

    test_dir = os.path.join(os.getcwd(), 'data', 'nabirds', 'images', 'test')
    if not os.path.isdir(test_dir):
        print("Testing folder does not exist. Adding it now. You can find it at {0}".format(test_dir))
        os.mkdir(test_dir)
    else:
        message = """
        Test folder already exists. \nWould you like to remove it and create a new one? (y/n)
        """

        test_resp = input(message)
        test_resp = test_resp.lower()

        if test_resp == 'y':
            os.remove(test_dir)
            os.mkdir(test_dir)
    
        
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

def plot_confusion_matrix(confusion_matrix, 
                            classes, 
                            normalize=False, 
                            title='Confusion Matrix', 
                            cmap=plt.cm.Blues):

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = confusion_matrix.max() / 2.

    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_val_images(dir_):

    # Get list of sub dirs
    sub_dirs = []
    for (f1,f2,f3) in os.walk(dir_):
        sub_dirs.append(f1)
    sub_dirs = sub_dirs[1:]

    # Get classes
    dict_ = {}
    with open('data/nabirds/classes.txt', 'r') as file_reader:
        classes_list = file_reader.readlines()
    
    for row in classes_list:
        row_list = row.split()
        dict_[int(row_list[0])] = row_list[1]

    import matplotlib.image as mpimg

    class_list = []
    img_list = []
    for sub_dir in sub_dirs:
        for (_, _, img_ids) in os.walk(sub_dir):
            pass

        for img_id in img_ids:
            img_name = os.path.join(sub_dir, img_id)

            cls_name = int(sub_dir.split('/')[-1])
            class_list.append(dict_[cls_name])

            img = mpimg.imread(img_name)
            img_list.append(img)

    return class_list, img_list, dict_


def predict_class(model, img):
    new_img = cv2.resize(img, (227,227))
    new_img_reshaped = new_img.reshape((1,227,227,3))
    output = model.predict(new_img_reshaped)
    output = np.argmax(output.round())
    return output

from keras import backend as K
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))