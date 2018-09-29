from helper_functions import split_train_test_dir, load_data
import numpy as np
import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(description='Arguments for bird training')
parser.add_argument('-gpu_id', '--GPU_IDs', type=list, required=True,help='The ids of the gpus being used as a string. \nEx: For gpus 0, 1, 2\n\tpython train_keras.py -gpu_id 012')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(args.GPU_IDs)
#-----------------------------------------------------------------------------------------------#
# Import Data
print('Build a .h5 file for the training instead of pulling from the directory during training.')

# Split data (train/dev/test, 70/15/15)
print('The dev set will be taken from the test set')
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
model = VGG16(weights='imagenet', include_top=False, input_shape=(299,299,3), classes=class_count)

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
# Compile
from keras import metrics
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['accuracy']
)

# Callback function (save best model only)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_best_{0}.hdf5'.format(message), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
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
    validation_data=(X_test, y_test),
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')

# Training Accuracy and Loss
if not os.path.isdir('history_data'):
    os.mkdir('history_data')

train_acc = history.history['acc']
train_loss = history.history['loss']
np.save('./history_data/train_acc_{0}.npy'.format(message), train_acc)
np.save('./history_data/train_loss_{0}.npy'.format(message), train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc_{0}.npy'.format(message), val_acc)
np.save('./history_data/val_loss_{0}.npy'.format(message), val_loss)

print('History saved!\n')
#-----------------------------------------------------------------------------------------------#