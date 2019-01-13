import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

import autokeras as ak
from helper_functions_autoML import get_data_base_path


# Necessary for autokeras to wrap the whole application in main func.
def main():

    # Set up training times
    TRAINING_TIMES = [
        60 * 60,		# 1 hour
        60 * 60 * 2,	# 2 hours
        60 * 60 * 4,	# 4 hours
        60 * 60 * 8,	# 8 hours
        60 * 60 * 12,	# 12 hours
        60 * 60 * 24,	# 24 hours
    ]
    
    # Import data
    BASE_DATA_DIR = get_data_base_path()
    DATASET_PATH = os.path.join(BASE_DATA_DIR, 'STL_10', 'data_STL_10.h5')

    # Read in data
    h5file = h5py.File(DATASET_PATH)
    X = np.array(h5file['X'])
    y = np.array(h5file['y'])
    y = y.reshape((len(y), 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    del X, y

    # Normalize
    X_train = X_train.astype("float") / 255.0
    X_test = X_test.astype("float") / 255.0  

    # Loop over the number of seconds to allow the current Auto-Keras 
    # model to train for.
    for seconds in TRAINING_TIMES:
        # Initialize model
        model = ak.ImageClassifier(verbose=True)

        # Fit to training data, first time
        model.fit(X_train, y_train, time_limit=seconds)

        # Fit again
        model.final_fit(X_train, y_train, X_test, y_test, retrain=True)

# if this is the main thread of execution then start the process (our
# code must be wrapped like this to avoid threading issues with
# TensorFlow)
if __name__ == "__main__":
    main()