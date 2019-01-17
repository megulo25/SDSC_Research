import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from autokeras.utils import pickle_from_file
from helper_functions_autoML import (get_data_base_path, getClassDict,
                                     interpretModelOutput)

# Argparser
arg = argparse.ArgumentParser()
arg.add_argument('-model', '--MODEL', required=True, help='Path to model')
args = vars(arg.parse_args())

# Paths
DATASET_BASE_PATH = get_data_base_path()
DATASET_PATH = os.path.join(DATASET_BASE_PATH, 'STL_10', 'data_STL_10.h5')
MODEL_PATH = os.path.join(os.getcwd(), str(args['MODEL']))

# Import Model
model = pickle_from_file(MODEL_PATH)

# Import Data
h5file = h5py.File(DATASET_PATH)
X = np.array(h5file['X'])
y = np.array(h5file['y'])
y_labels = np.array(h5file['y_labels'])

# Test

# Ground Truth
while True:
    random_number = np.random.randint(X.shape[0])
    img = X[random_number]
    img_plot = img
    img_plot = img_plot.astype('int')
    y_output = y[random_number]
    y_class_name = y_labels[random_number].decode('utf-8')

    # Model Output
    y_pred = model.predict(img)
    class_name = interpretModelOutput(y_pred)
    print(y_pred)
    print(class_name)

    # Plot Results
    plt.imshow(img_plot)
    plt.title('y_true: {0}\ny_pred: {1}'.format(y_class_name, class_name))
    plt.show()