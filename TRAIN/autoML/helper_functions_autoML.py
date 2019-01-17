# Import modules
from sklearn.preprocessing import LabelEncoder
from random import shuffle
import os

def get_data_base_path():
    current_dir = os.getcwd()
    current_dir_list = current_dir.split('/')
    base_wo_data = '/'.join(current_dir_list[:-2])
    base_w_data = os.path.join(base_wo_data, 'data', 'datasets')
    return base_w_data

def shuffleData(X, y):
    N = X.shape[0]
    indicies = [i for i in range(N)]
    shuffle(indicies)
    new_X = X[indicies]
    new_y = y[indicies]
    return new_X, new_y