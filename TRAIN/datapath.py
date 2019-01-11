# Pass the base diretory for the datasets
import os

def get_data_base_path():
    current_dir = os.getcwd()
    current_dir_list = current_dir.split('/')
    base_wo_data = '/'.join(current_dir_list[:-1])
    base_w_data = os.path.join(base_wo_data, 'data')
    return base_w_data