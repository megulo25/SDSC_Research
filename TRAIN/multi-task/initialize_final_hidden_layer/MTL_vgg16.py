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

# Load in data
print('Loading in data...')
a = os.getcwd()
b = a.split('/')
b = b[:-2]
c = '/'.join(b)
filename = os.path.join(c, 'data' , 'dataset_555','data_555.h5')

# Loading in the 555 class dataset for MTL
f = h5py.File(filename)
X_train = np.array(f['X_train'])
X_validation = np.array(f['X_test'])
y_train = np.array(f['y_train'])
y_validation = np.array(f['y_test'])

# Concatenate the data, we will split later
X = np.concatenate([X_train, X_validation])
y = np.concatenate([y_train, y_validation])

# Free up memory
del X_train
del X_validation
del y_train
del y_validation

print('Dataset loaded!\n')
#-----------------------------------------------------------------------------------------------#
# Split test into dev/test

# Split classification label from bounding box predictions
def label_split(y):
    y_new = []
    for i in y:
        # Classification, bounding box
        y_new.append((i[:-4], i[-4:]))
    return y_new

# Split Data to training/validation (80%, 20%)
test_split = .2
print('Splitting dataset: {0} training, {1} validation'.format(1-test_split, test_split))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split)

# Free up memory
del X
del y

# Separate classification output and bounding box output
new_y_train = label_split(y_train)
new_y_val = label_split(y_val)


print('Data:')
print('Training set: {0}'.format(X_train.shape))
print('Test set: {0}'.format(X_val.shape))
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
output_layer_class_classification = Dense(class_count-4, activation='softmax', name='class_classification')(x_class_classification)

# Multi-output (bounding box)
x_bounding_box = model.output
x_bounding_box = Flatten()(x_bounding_box)
output_layer_bounding_box = Dense(4, activation='linear', name='bounding_box')(x_bounding_box)


model = Model(inputs=model.input, outputs=[output_layer_class_classification, output_layer_bounding_box])
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
# Compile
from keras import metrics
model.compile(
    loss={
        'class_classification': 'binary_crossentropy',
        'bounding_box': 'mse'
    },
    optimizer='adam'
)

# Callback function (save best model only)
if not os.path.isdir('models'):
    os.mkdir('models')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./models/model_multi_task_best_{0}.hdf5'.format(model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')
batchsize = 16

y_train_classification, y_train_bounding_box = zip(*new_y_train)
y_train_classification = np.array(y_train_classification)
y_train_bounding_box = np.array(y_train_bounding_box)

y_dev_classification, y_dev_bounding_box = zip(*new_y_val)

history = model.fit(
    {
        'input_1': X_train
    },
    {
        'class_classification': y_train_classification,
        'bounding_box': y_train_bounding_box
    },
    batch_size=batchsize,
    epochs=2000,
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