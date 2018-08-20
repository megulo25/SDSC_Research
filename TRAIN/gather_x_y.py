from helper_functions import train_test_split_multi_output
import os

cur_dir = os.getcwd()
full_path = os.path.join(cur_dir, 'data', 'nabirds', 'images')

train_test_split_multi_output(full_path)