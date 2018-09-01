from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np

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
confusion_matrix_values = []
for i in range(cnf_mat.shape[0]):
    confusion_matrix_values.append(cnf_mat[i])

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

trace = {
    "x":class_list,
    "y":class_list,
    "z":confusion_matrix_values,
    "colorscale": "RdYlBu",
    "showscale": True,
    "type": "heatmap"
}

# Save Trace
path_to_data_folder_in_webapp = os.getcwd()
path_to_data_folder_in_webapp = path_to_data_folder_in_webapp.split('/')[:-1]
path_to_data_folder_in_webapp = os.path.join(path_to_data_folder_in_webapp, 'web_app', 'static', 'data')
import json
trace_full_path = os.path.join(path_to_data_folder_in_webapp, 'trace.json')
with open(trace_full_path, 'w') as trace_writer:
    json.dump(trace, trace_writer)

# Save data_list
data_list_full_path = os.path.join(path_to_data_folder_in_webapp, 'data_list.json')
with open(data_list_full_path, 'w') as data_writer:
    json.dump(dict_list, data_writer)