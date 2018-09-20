from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#-----------------------------------------------------------------------------------------------#
# Load in data

a = os.getcwd()
b = a.split('/')
b = b[:-1]
c = '/'.join(b)
filename = os.path.join(c, 'data' ,'data_10.h5')

f = h5py.File(filename, 'r')
X = np.array(f['X'])
y = np.array(f['y'])
#-----------------------------------------------------------------------------------------------#
# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
del X
del y
#-----------------------------------------------------------------------------------------------#
# Apply Image Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zca_whitening=True,
    rotation_range=90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=.2,
    height_shift_range=.2
    )

datagen.fit(X_train)
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = y_train.shape[1]

# Import InceptionNet
print('Loading in model...')
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model


model = VGG16(weights='imagenet', include_top=False, input_shape=(299,299,3), classes=class_count)
# model = ResNet50(weights='imagenet', include_top=False,input_shape=(299,299,3), classes=class_count)
# model = InceptionV3(weights='imagenet', include_top=False,input_shape=(299,299,3), classes=class_count)

# Get model name
model_name = model.name

x = model.output
x = Flatten()(x)
output_layer = Dense(class_count, activation='softmax')(x)
model = Model(inputs=model.input, outputs=output_layer)
print('Model loaded!\n')

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
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_multi_task_best_{0}.hdf5'.format(model_name), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')

history = model.fit_generator(
    datagen.flow(
        x=X_train,
        y=y_train,
        batch_size=16
    ),
    steps_per_epoch= len(X_train) // 16,
    epochs=200,
    verbose=1,
    validation_data=(X_test, y_test),
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
np.save('./history_data/train_acc_{0}.npy'.format(model_name), train_acc)
np.save('./history_data/train_loss_{0}.npy'.format(model_name), train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc_{0}.npy'.format(model_name), val_acc)
np.save('./history_data/val_loss_{0}.npy'.format(model_name), val_loss)

print('History saved!\n')