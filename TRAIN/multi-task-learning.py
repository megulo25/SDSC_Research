from helper_functions import multitask_loss
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os
#-----------------------------------------------------------------------------------------------#
# Load in data

filename = 'data.h5'
f = h5py.File(filename, 'r')
X = np.array(f['X'])
y = np.array(f['y'])
#-----------------------------------------------------------------------------------------------#
# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#-----------------------------------------------------------------------------------------------#
# Import Model

# Import InceptionNet
print('Loading in model...')
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3), classes=class_count)
x = model.output
x = Flatten()(x)
output_layer = Dense(class_count, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model loaded!\n')

# Multi-gpu
from keras.utils import multi_gpu_model
model = multi_gpu_model(model)

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Image Pre-processing
from keras.preprocessing.image import ImageDataGenerator

# Training Generator
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    rotation_range=20,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    zca_epsilon=1e-6,
                                    fill_mode="nearest")

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(X_train)

train_generator = train_datagen.flow(
    x=X_train,
    y=y_train,
    batch_size=64,
    shuffle=True
)

validation_datagen = ImageDataGenerator(
    rescale=1
)

validation_generator = validation_datagen.flow(
    x=X_test,
    y=y_test,
    batch_size=1,
    shuffle=True
)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss=multitask_loss,
    optimizer='SGD',
    metrics=['accuracy']
)

# Callback function (save best model only)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_multi_task_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')
history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    epochs=200,
    verbose=2,
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save Model and Training Process
print('Saving history...')

if not os.path.isdir('history_data'):
    os.mkdir('history_data')

train_acc = history.history['acc']
train_loss = history.history['loss']
np.save('./history_data/train_acc.npy', train_acc)
np.save('./history_data/train_loss.npy', train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc.npy', val_acc)
np.save('./history_data/val_loss.npy', val_loss)

print('History saved!\n')