# Complete
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import h5py
import os

#-----------------------------------------------------------------------------------------------#
# Arg parser
arg = argparse.ArgumentParser()
arg.add_argument('-gpu_id', '--gpu_id', required=True, help='ID of GPU')
args = vars(arg.parse_args())
#-----------------------------------------------------------------------------------------------#
os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])
filename = os.path.join(os.getcwd(), 'data_10.h5')

# Load in data
print('Loading in data...')
f = h5py.File(filename)
X = np.array(f['X'])
y = np.array(f['y'])
print('Dataset loaded!\n')
#-----------------------------------------------------------------------------------------------#
# Split test into dev/test
test_split = .2
print('Splitting dataset: {0} training, {1} validation'.format(1-test_split, test_split))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split)

# Free up memory
del X
del y
print('Dataset split!\n)')
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = y_train.shape[1]

# Import Model
print('Loading in model...')
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

## Models

# VGG 16
from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3), classes=class_count)

# Get model name
model_name = model.name

# Multi-output (class classification, one vs. rest)
x_class_classification = model.output
x_class_classification = Flatten()(x_class_classification)
output_layer_class_classification = Dense(class_count, activation='softmax', name='class_classification')(x_class_classification)


model = Model(inputs=model.input, outputs=[output_layer_class_classification])
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
# Compile
from keras import metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_multi_task_best_{0}.hdf5'.format(model_name), monitor='val_acc', verbose=1, save_best_only=True)
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')
batchsize = 16

history = model.fit(
    x=X_train,
    y=y_train
    batch_size=batchsize,
    epochs=200,
    verbose=1,
    validation_split=.25,
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Evaluate on test set
#-----------------------------------------------------------------------------------------------#
# Save Model and Training Process
print('\nSaving history...')
np.save('history_{0}.npy'.format(model_name), history)
print('History saved!\n')