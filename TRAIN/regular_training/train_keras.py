import numpy as np
import os
import sys
import shutil
import argparse
import h5py

parser = argparse.ArgumentParser(description='Arguments for bird training')
parser.add_argument('-gpu_id', '--GPU_IDs', type=list, required=True,help='The ids of the gpus being used as a string. \nEx: For gpus 0, 1, 2\n\tpython train_keras.py -gpu_id 012')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(args.GPU_IDs)
#-----------------------------------------------------------------------------------------------#
# Import Data
print('Loading in data...')
data_full_path = os.path.join('/'.join(os.getcwd().split('/')[:-1]), 'data', 'data_555.h5')
file = h5py.File(data_full_path)
X_train = np.array(file['X_train'])
X_val = np.array(file['X_test'])
y_train = np.array(file['y_train'])
y_val = np.array(file['y_test'])
print('Data loaded!\n')

# Exclude bounding box values
y_train = y_train[:, :555]
y_val = y_val[:, :555]

# Split data (train/dev/test, 70/15/15)
print('The dev set will be taken from the test set')
from sklearn.model_selection import train_test_split
X_dev, X_test, y_dev, y_test = train_test_split(X_val, y_val, test_size=.5, shuffle=True)
print('Dataset split!\n')
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = y_train.shape[1]

print('Loading in model...')
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

## Models

# VGG 16
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3), classes=class_count)

# Get model name
model_name = model.name

x = model.output
x = Flatten()(x)
output_layer = Dense(class_count, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Image preprocessing:
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True
)

datagen.fit(X_train)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']
)

# Callback function (save best model only)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_best_{0}.hdf5'.format(model_name), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train
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
    validation_data=(X_dev, y_dev),
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')

# Training Accuracy and Loss
if not os.path.isdir('history_data'):
    os.mkdir('history_data')

# Save history object
np.save('./history_data/history_{0}'.format(model_name))

print('History saved!\n')
#-----------------------------------------------------------------------------------------------#