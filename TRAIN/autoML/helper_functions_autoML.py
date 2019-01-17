# Import modules
from sklearn.preprocessing import LabelEncoder
import os

def get_data_base_path():
    current_dir = os.getcwd()
    current_dir_list = current_dir.split('/')
    base_wo_data = '/'.join(current_dir_list[:-2])
    base_w_data = os.path.join(base_wo_data, 'data', 'datasets')
    return base_w_data

def getClassDict(CLASSES_TXT_PATH):
    class_dict = {}
    # Read in classes.txt
    with open('{0}'.format(CLASSES_TXT_PATH), 'r') as file_reader:
        class_txt = file_reader.readlines()
    for _, value in enumerate(class_txt):
        # Split
        value_split = value.split()
        number = value_split[0]
        # Rejoin
        value_rejoined = ' '.join(value_split[1:])
        # Create temp dict
        dict_ = {
            number: value_rejoined
        }
        # Append to dict
        class_dict = {**class_dict, **dict_}
    return class_dict

def interpretModelOutput(y):
    "This will only return the class name of one prediction"
    
    BASE_DIR = get_data_base_path()
    DATASET_555 = os.path.join(BASE_DIR, 'datasets', 'dataset_555', 'images')
    CLASSES_TXT_PATH = os.path.join(BASE_DIR, 'datasets', 'dataset_555', 'classes.txt')
    labels = os.listdir(DATASET_555)

    label_encoder = LabelEncoder()
    subdir_encodings = label_encoder.fit_transform(labels)
    subdir_encodings = list(subdir_encodings)

    # Get class dict
    class_dict = getClassDict(CLASSES_TXT_PATH)

    # Convert output to list
    list_ = list(y)

    # Search for 1
    temp_idx = list_.index(1)

    # Take temp index to search for real index value
    idx = subdir_encodings.index(temp_idx)

    # Use index to get parent directory name (e.g. 0295)
    subdir = labels[idx]

    # Search through class_dict to get class name
    class_name = class_dict[str(int(subdir))]
    return class_name