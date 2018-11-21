from sklearn.model_selection import train_test_split
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
        list_ = os.listdir()
        os.removedirs(list_[1])
        os.removedirs(list_[5])
        os.removedirs(list_[7])
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