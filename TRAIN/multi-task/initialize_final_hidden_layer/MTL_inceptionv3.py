# Complete
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#-----------------------------------------------------------------------------------------------#
# Load in data
print('Loading in data...')
a = os.getcwd()
b = a.split('/')
b = b[:-2]
c = '/'.join(b)
filename = os.path.join(c, 'data' , 'dataset_555','data_555_MTL.h5')

# Loading in the 555 class dataset for MTL
f = h5py.File(filename)
X = np.array(f['X'])
y = np.array(f['y'])

print('Dataset loaded!\n')
#-----------------------------------------------------------------------------------------------#
# Split dataset
test_split = .3
print('Splitting dataset: {0} training, {1} validation'.format(1-test_split, test_split))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
del X
del y
print('Dataset split!\n)')
#-----------------------------------------------------------------------------------------------#
# Apply Image Data Augmentation
print('Applying Data augmentation...')
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zca_whitening=False,
    rotation_range=90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=.2,
    height_shift_range=.2
    )

datagen.fit(X_train)
print('Data Augmentation complete!')
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = y_train.shape[1]

# Import InceptionNet
print('Loading in model...')
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

## Models

# VGG 16
# from keras.applications.vgg16 import VGG16
# model = VGG16(weights='imagenet', include_top=False, input_shape=(299,299,3), classes=class_count)

# Resnet 50
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet', include_top=False,input_shape=(299,299,3), classes=class_count)

# InceptionV3
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3(weights='imagenet', include_top=False,input_shape=(299,299,3), classes=class_count)

# Get model name
model_name = model.name

x = model.output
x = Flatten()(x)
output_layer = Dense(class_count, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model {0} loaded!\n'.format(model_name))

# Create the final hidden layer
def create_final_hidden_layer():
    # Hierarchy file
    a = os.getcwd()
    b = a.split('/')
    b = b[:-2]
    c = '/'.join(b)
    hierarchy_dict = np.load(os.path.join(c, 'data', 'hierarchy_dict.npy')).item()
    # Get a list of all the classes 
    path = os.path.join(c, 'data', 'nabirds_555', 'nabirds', 'images')
    list_ = []
    for (_, l, _) in os.walk(path):
        list_.append(l)
    list_ = list_[0]
    # Create array
    final_hidden_layer = np.zeros((len(list_), 1011))
    # Get the heirarchy for each class and place into array
    c = 0
    for i, j in enumerate(list_):
        # Convert to integer
        j = int(j)
        # Get hierarchy 
        y = list(hierarchy_dict[j])
        # Index array
        final_hidden_layer_old = final_hidden_layer[i]
        final_hidden_layer[i][y] = 1
    return final_hidden_layer

# Initialize weights in final hidden layer
def initialze_final_hidden_layer(weights, array_of_all_hierarchies_in_training_set, flag='init_a'):
    # Grab the weights from the last hidden layer
    last_hidden_layer_weights = weights[-2]
    n = len(array_of_all_hierarchies_in_training_set)
    # Replace the first n rows with possible hierarchies
    last_hidden_layer_weights[:n] = array_of_all_hierarchies_in_training_set
    # Initialize the rest of the rows with one of the following
    if (flag == 'init_a'):
        # Initialize the rest with zeros
        na = last_hidden_layer_weights[n:].shape
        last_hidden_layer_weights[n:] = np.zeros(na)
    elif (flag == 'init_b'):
        # Initialize the rest with random values.
        nb = last_hidden_layer_weights[n:].shape
        last_hidden_layer_weights[n:] = np.random.rand(nb)
    # Replace last hidden layer in weights list
    weights[-2] = last_hidden_layer_weights
    return weights

print('Initializing weights...')
weights = model.get_weights()
array_of_all_hierarchies_in_training_set = create_final_hidden_layer()
new_weights = initialze_final_hidden_layer(weights=weights, array_of_all_hierarchies_in_training_set=array_of_all_hierarchies_in_training_set)

# Set new weights to the model
model.set_weights(new_weights)

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
# Binary Cross-Entropy Loss (Logistic Loss)
"""
The only way to use this loss function properly is to make sure that the accuracy is 
defined as categorical accuracy. We must call this accuracy from the keras.metrics
library.
"""
# Compile
from keras import metrics
from keras.metrics import categorical_accuracy
model.compile(
    loss='binary_crossentropy',
    optimizer='SGD',
    metrics=[categorical_accuracy]
)

# Callback function (save best model only)
if not os.path.isdir('models'):
    os.mkdir('models')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./models/model_multi_task_best_{0}_inita.hdf5'.format(model_name), monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')
batchsize = 16

history = model.fit_generator(
    datagen.flow(
        x=X_train,
        y=y_train,
        batch_size=batchsize
    ),
    steps_per_epoch= len(X_train) // batchsize,
    epochs=500,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save Model and Training Process
print('Saving history...')

if not os.path.isdir('./models/history_data'):
    os.mkdir('./models/history_data')

train_acc = history.history['categorical_accuracy']
train_loss = history.history['loss']
np.save('./models/history_data/train_acc_{0}.npy'.format(model_name), train_acc)
np.save('./models/history_data/train_loss_{0}.npy'.format(model_name), train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_categorical_accuracy']
val_loss = history.history['val_loss']
np.save('./models/history_data/val_acc_{0}.npy'.format(model_name), val_acc)
np.save('./models/history_data/val_loss_{0}.npy'.format(model_name), val_loss)

print('History saved!\n')