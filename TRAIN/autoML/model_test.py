import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from autokeras.utils import pickle_from_file
from keras.utils import plot_model
from helper_functions_autoML import get_data_base_path

# Argparser
arg = argparse.ArgumentParser()
arg.add_argument('-model', '--MODEL', required=True, help='Path to model')
args = vars(arg.parse_args())

# Paths
DATASET_BASE_PATH = get_data_base_path()
DATASET_PATH = os.path.join(DATASET_BASE_PATH, 'STL_10', 'data_STL_10.h5')
DATASET_10_PATH = os.path.join(DATASET_BASE_PATH, 'dataset_10')
MODEL_PATH = os.path.join(os.getcwd(), str(args['MODEL']))

# Plot output
PLOTS_PATH = os.path.join(os.getcwd(), 'OUTPUT', 'model_architecture_plots')
if not os.path.isdir(PLOTS_PATH):
    os.mkdir(PLOTS_PATH)

# Help get class name
dataset_10_list = os.listdir(DATASET_10_PATH)
class_dict = {
    0: dataset_10_list[0],
    1: dataset_10_list[1],
    2: dataset_10_list[2],
    3: dataset_10_list[3],
    4: dataset_10_list[4],
    5: dataset_10_list[5],
    6: dataset_10_list[6],
    7: dataset_10_list[7],
    8: dataset_10_list[8],
    9: dataset_10_list[9]
}

# Import Model
model = pickle_from_file(MODEL_PATH)

# Plot model
model_name = MODEL_PATH.split('/')
model_name = model_name[-1]
model_name = model_name[:-3]
keras_model = model.graph.produce_keras_model()
MODEL_ARCH_FULL_PATH = os.path.join(PLOTS_PATH, '{0}.png'.format(model_name))
plot_model(keras_model, to_file=MODEL_ARCH_FULL_PATH)

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
    img_shape = img.shape
    img = img.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
    img_plot = img_plot.astype('int')
    y_output = y[random_number]
    y_class_name = y_labels[random_number].decode('utf-8')

    # Model Output
    y_pred = model.predict(img)
    y_pred = y_pred[0]

    # Get class name
    class_name = class_dict[y_pred]

    # Plot Results
    plt.imshow(img_plot)
    plt.title('y_true: {0}\ny_pred: {1}'.format(y_class_name, class_name))
    plt.show()