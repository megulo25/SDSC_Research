from helper_functions import multitask_loss
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#-----------------------------------------------------------------------------------------------#
# Load in data

filename = 'data/data_10.h5'
f = h5py.File(filename, 'r')
X = np.array(f['X'])
y = np.array(f['y'])
#-----------------------------------------------------------------------------------------------#
# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
del X
del y
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = y_train.shape[1]

# Import InceptionNet
print('Loading in model...')
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model

model = VGG16(weights='imagenet', input_shape=(299,299,3), classes=class_count)
model = VGG19(weights='imagenet', input_shape=(299,299,3), classes=class_count)
model = ResNet50(weights='imagenet', input_shape=(299,299,3), classes=class_count)
model = InceptionV3(weights='imagenet', input_shape=(299,299,3), classes=class_count)

model_name = 'vgg16'
print('Model loaded!\n')

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
from keras import losses
squared_hinge = losses.squared_hinge
categorical_hinge = losses.categorical_hinge
categorical_cross_entropy = losses.categorical_crossentropy
sparse_categorical_crossentropy = losses.sparse_categorical_crossentropy

loss_function = multitask_loss
loss_name = 'multitask_loss'

# Compile
from keras import metrics
model.compile(
    loss=loss_function,
    optimizer='adam',
    metrics=['accuracy']
)

# Callback function (save best model only)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_multi_task_best_{0}_{1}.hdf5'.format(model_name,loss_name), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=16,
    epochs=500,
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
np.save('./history_data/train_acc_{0}.npy'.format(loss_name), train_acc)
np.save('./history_data/train_loss_{0}.npy'.format(loss_name), train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc_{0}.npy'.format(loss_name), val_acc)
np.save('./history_data/val_loss_{0}.npy'.format(loss_name), val_loss)

print('History saved!\n')