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


#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
cnf_mat = confusion_matrix(validation_generator.classes, y_pred)

confusion_matrix_values = []
for i in range(cnf_mat.shape[0]):
    confusion_matrix_values.append(cnf_mat[i])

print('Classification Report')
target_names = list(validation_generator.class_indices.keys())
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
# print(class_list)
# print(cnf_mat)

# List of dictionaries
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

# Save

# Show the training
# import matplotlib.pyplot as plt

# Load in historical data
# n = 0
# train_acc = np.load('history_data/train_acc_{0}.npy'.format(n))
# val_acc = np.load('history_data/val_acc_{0}.npy'.format(n))

# train_loss = np.load('history_data/train_loss_{0}.npy'.format(n))
# val_loss = np.load('history_data/val_loss_{0}.npy'.format(n))

# # Plot Accuracy
# num_iters = np.arange(len(train_acc))
# plt.figure()
# plt.plot(num_iters, train_acc, 'g', label='Training')
# plt.plot(num_iters, val_acc, 'b', label='Validation')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Accuracy (%)')
# plt.title('Training set vs Validation')
# plt.grid()
# plt.legend()
# # plt.show()

# # Plot Loss
# num_iters = np.arange(len(train_loss))
# plt.figure()
# plt.plot(num_iters, train_loss, 'g', label='Training')
# plt.plot(num_iters, val_loss, 'b', label='Validation')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Loss')
# plt.title('Training set vs Validation')
# plt.grid()
# plt.legend()
# plt.show()


# Prettier confusion matrix
import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('meguloabebe', 'P5vXd5GcQr07emsSeLul')
trace = {
    "x":class_list,
    "y":class_list,
    "z":confusion_matrix_values,
    "colorscale": "RdYlBu",
    "showscale": True,
    "type": "heatmap"
}
data = Data([trace])
layout = {
    "annotations":dict_list
}
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)