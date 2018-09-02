from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import json

validation_dir = os.path.join(os.getcwd(), 'data', 'nabirds_9', 'test')

# Import model
model = load_model('model_best.hdf5')

# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
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
Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cnf_mat = confusion_matrix(validation_generator.classes, y_pred)

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
trace = {
    "x":class_list,
    "y":class_list,
    "z":cnf_mat_jsonified,
    "colorscale": "RdYlBu",
    "showscale": True,
    "type": "heatmap"
}

# Save Trace
path_to_data_folder_in_webapp = os.getcwd()
path_to_data_folder_in_webapp = path_to_data_folder_in_webapp.split('/')[:-1]
path_to_data_folder_in_webapp = "/".join(path_to_data_folder_in_webapp)
path_to_data_folder_in_webapp = os.path.join(path_to_data_folder_in_webapp, 'web_app', 'static', 'data')

# Bring in training process
train_acc = np.load('history_data/train_acc_0.npy')
val_acc = np.load('history_data/val_acc_0.npy')
train_loss = np.load('history_data/train_loss_0.npy')
val_loss = np.load('history_data/val_loss_0.npy')

# Save data
dict_ = {
    'trace':trace,
    'data_list':dict_list,
    'train_acc':train_acc,
    'train_loss':train_loss,
    'val_acc':val_acc,
    'val_loss':val_loss
}

filename = os.path.join(path_to_data_folder_in_webapp, 'data.json')
with open(filename, 'w') as file_writer:
    json.dump(dict_, file_writer)