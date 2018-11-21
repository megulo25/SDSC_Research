import numpy as np
import os
import sys
import shutil
import argparse
import h5py
from helper_function import *

parser = argparse.ArgumentParser(description='Arguments for bird training')
parser.add_argument('-gpu_id', '--GPU_IDs', type=list, required=True, help='The ids of the gpus being used as a string. \nEx: For gpus 0, 1, 2\n\tpython train_keras.py -gpu_id 012')
parser.add_argument('-split', '--split', type=int, required=True, help='Split dataset. 0=no, 1=yes')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(args.GPU_IDs)
#-----------------------------------------------------------------------------------------------#
# Import Data
print('Loading in data...')
check_data_folder()
check_nabirds10_folder()

# Split dataset
test_split = .3

if int(args.split) == 1:
    split_data(test_split)
    
print('Data loaded!\n')
#-----------------------------------------------------------------------------------------------#
# Import Model
print('Loading in model...')
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

## Models

# VGG 16
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Get model name
model_name = model.name

x = model.output
x = Flatten()(x)
output_layer = Dense(10, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Image preprocessing:
from keras.preprocessing.image import ImageDataGenerator

# Train
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=.2,
    height_shift_range=.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/nabirds10/train',
    target_size = (224, 224),
    batch_size=16
)

# Validation
val_datagen = ImageDataGenerator(
    rescale=1
)

val_generator = val_datagen.flow_from_directory(
    'data/nabirds10/validation',
    target_size=(224,224),
    batch_size=2
)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss='binary_crossentropy',
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
batchsize = 32

history = model.fit_generator(
    train_generator,
    epochs=200,
    verbose=1,
    validation_data=val_generator,
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')

# Save history object
np.save('history_{0}'.format(model_name), history)

print('History saved!\n')
#-----------------------------------------------------------------------------------------------#