import os
from shutil import copyfile
from random import sample
from math import floor


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