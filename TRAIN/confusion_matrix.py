from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import os
import numpy as np
import json
import cv2

dir_name = 'nabirds_10'
validation_dir = os.path.join(os.getcwd(), 'data', dir_name, 'test', 'test')

# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1
)

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(299,299),
    batch_size=1,
    class_mode='categorical'
)
class_indicies = validation_generator.class_indices
class_list = list(class_indicies.keys())

#Confusion Matrix
true_class = validation_generator.classes

# Import model
from keras.models import load_model
model = load_model('model_best_{0}.hdf5'.format(dir_name))

# Get a list of the images
num_classes = len(class_indicies)
y_pred_array = np.zeros((1,num_classes))
base_validation_path = os.path.join(os.getcwd(), 'data', dir_name, 'test')
for n in range(len(validation_generator.filenames)):
    img_path = os.path.join(base_validation_path, validation_generator.filenames[n])
    img = mpimg.imread(img_path)
    img = cv2.resize(img, (299, 299))
    img = np.reshape(img, (1,299,299,3))
    y_pred_ex = model.predict(img)
    y_pred_ex = y_pred_ex.round()
    y_pred_array = np.concatenate([y_pred_array, y_pred_ex])

y_pred_array = y_pred_array[1:]

def confusion_matrx(true_class, y_pred):
    cnf_mat = np.zeros((y_pred.shape[1], y_pred.shape[1]))
    for i in range(y_pred.shape[0]):
        cnf_mat[true_class[i], y_pred[i].argmax()] += 1
    return cnf_mat

cnf_mat = confusion_matrx(true_class, y_pred_array)

# List of dictionaries for plotly
dict_list = []
for j in range(cnf_mat.shape[0]):
    for k in range(cnf_mat.shape[1]):
        dict_ = {
            "x":class_list[j],
            "y":class_list[k],
            "font":{"color":"black"},
            "showarrow":False,
            "text":str(cnf_mat[j,k]),
        }

        dict_list.append(dict_)

cnf_mat_jsonified = cnf_mat.tolist()
trace_data = {
    "x":class_list,
    "y":class_list,
    "z":cnf_mat_jsonified,
    "colorscale": "RdYlBu",
    "showscale": True,
    "type": "heatmap"
}

trace_layout = {
    "annotations":dict_list,
    "yaxis":{
        'showticklabels':True
    }
}

# Save Trace
path_to_data_folder_in_webapp = os.getcwd()
path_to_data_folder_in_webapp = path_to_data_folder_in_webapp.split('/')[:-1]
path_to_data_folder_in_webapp = "/".join(path_to_data_folder_in_webapp)
path_to_data_folder_in_webapp = os.path.join(path_to_data_folder_in_webapp, 'web_app', 'static', 'data')

# Create data folder, if does not exist
if not os.path.isdir(path_to_data_folder_in_webapp):
    os.mkdir(path_to_data_folder_in_webapp)

# Bring in training process
train_acc = np.load('history_data/train_acc_{0}.npy'.format(dir_name)).tolist()
val_acc = np.load('history_data/val_acc_{0}.npy'.format(dir_name)).tolist()
train_loss = np.load('history_data/train_loss_{0}.npy'.format(dir_name)).tolist()
val_loss = np.load('history_data/val_loss_{0}.npy'.format(dir_name)).tolist()


# Save data
dict_ = {
    'trace_data':trace_data,
    'trace_layout':trace_layout,
    'train_acc':train_acc,
    'train_loss':train_loss,
    'val_acc':val_acc,
    'val_loss':val_loss
}

filename = os.path.join(path_to_data_folder_in_webapp, 'data.json')
with open(filename, 'w') as file_writer:
    json.dump(dict_, file_writer)