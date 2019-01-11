from ..datapath import get_data_base_path
from helper_functions_autoML import *
import autokeras as ak
import os

# Necessary for autokeras to wrap the whole application in main func.
def main():

    # Set up training times
    n_sec = 3600
    TRAINING_TIMES = [2**i * n_sec for i in range(5)]
    
    # Import data
    BASE_DATA_DIR = get_data_base_path()

    # Normalize X

    # Split data

    # Initialize class labels
    labels = ['western_grebe']

    # Loop over the number of seconds to allow the current Auto-Keras 
    # model to train for.

    for seconds in TRAINING_TIMES:
        # Initialize model
        model = ak.ImageClassifier(verbose=True)

        # Fit to training data, first time
        model.fit(X_train, y_train, time_limit=seconds)

        # Fit again
        model.final_fit(X_train, y_train, X_test, y_test, retrain=True)

        # Evaluate model

        # Write report to text file




# if this is the main thread of execution then start the process (our
# code must be wrapped like this to avoid threading issues with
# TensorFlow)
if __name__ == "__main__":
    main()