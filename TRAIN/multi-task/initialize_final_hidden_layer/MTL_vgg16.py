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
b = b[:-1]
c = '/'.join(b)
filename = os.path.join(c, 'data' , 'dataset_555','data_555.h5')

# Loading in the 555 class dataset for MTL
f = h5py.File(filename)
X_train = np.array(f['X_train'])
X_validation = np.array(f['X_test'])
y_train = np.array(f['y_train'])
y_validation = np.array(f['y_test'])

print('Dataset loaded!\n')
#-----------------------------------------------------------------------------------------------#
# Split test into dev/test
test_split = .5
print('Splitting dataset: {0} training, {1} validation'.format(1-test_split, test_split))
X_dev, X_test, y_dev, y_test = train_test_split(X_validation, y_validation, test_size=test_split, random_state=42)

print('Data:')
print('Training set: {0}'.format(X_train.shape))
print('Dev set: {0}'.format(X_dev.shape))
print('Test set: {0}'.format(X_test.shape))
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

x = model.output
x = Flatten()(x)
output_layer = Dense(class_count, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
# Compile
from keras import metrics
model.compile(
    loss='categorical_crossentropy',
    optimizer='SGD',
    metrics=['categorical_accuracy']
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

history = model.fit_generator(
    datagen.flow(
        x=X_train,
        y=y_train,
        batch_size=batchsize
    ),
    steps_per_epoch= len(X_train) // batchsize,
    epochs=200,
    verbose=1,
    validation_data=(X_dev, y_dev),
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Evaluate on test set
from sklearn.metrics import accuracy_score, log_loss
y_pred = model.predict(X_test)
y_true = y_test

acc_class = log_loss(y_true[:-4], y_pred[:-4])
acc_bbox = accuracy_score(y_true[-4:], y_pred[-4:])

print('Class accuracy on test set: {0]'.format(acc))
print('Bounding box accuracy: {0}'.format(1-acc_bbox))
#-----------------------------------------------------------------------------------------------#
# Save Model and Training Process
print('Saving history...')
np.save('history_{0}.npy'.format(model_name))
print('History saved!\n')