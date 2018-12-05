# Complete
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import argparse
import h5py
import os

#-----------------------------------------------------------------------------------------------#
# Arg parser
arg = argparse.ArgumentParser()
arg.add_argument('-gpu_id', '--gpu_id', required=True, help='ID of GPU')
arg.add_argument('-o', '--optimizer', required=True, help='select optimizer')
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
# Split data into train/dev
test_split = .2
print('Splitting dataset: {0} training, {1} development'.format(1-test_split, test_split))
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=test_split)

# Split test set into dev/val
X_test, X_val, y_test, y_val = train_test_split(X_dev, y_dev, test_size=.5)

# Free up memory
del X
del y
del X_dev
del y_dev
print('Dataset split!\n)')
#-----------------------------------------------------------------------------------------------#
# Split output

# Val set
y_leaf_val = y_val[:, 5:]
y_higher_val = y_val[:, :5]

# test set
y_leaf_test = y_test[:, 5:]
y_higher_test = y_test[:, :5]

del y_test
del y_val
#-----------------------------------------------------------------------------------------------#
# Preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=20.0
)
train_datagen.fit(X_train)
#-----------------------------------------------------------------------------------------------#
# Import Model
class_count = 15

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
output_layer_class_10_classification = Dense(10, activation='softmax', name='leaf_nodes')(x_class_classification)
output_layer_class_5_classification = Dense(5, activation='softmax', name='higher_nodes')(x_class_classification)


model = Model(inputs=model.input, outputs=[output_layer_class_10_classification, output_layer_class_5_classification])
print('Model {0} loaded!\n'.format(model_name))

# Output Model Summary
model.summary()
#-----------------------------------------------------------------------------------------------#
# Loss
# Compile
losses = {
    'leaf_nodes':'binary_crossentropy',
    'higher_nodes':'binary_crossentropy'
}

from keras import metrics
model.compile(
    loss=losses,
    optimizer=str(args['optimizer']),
    metrics={
        'leaf_nodes':'acc',
        'higher_nodes':'acc'
    }
)

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./models/model_multi_task_best_{0}_{1}_gen.hdf5'.format(model_name, str(args['optimizer'])), monitor='val_acc', verbose=1, save_best_only=False)
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Data batch Generator
def batch_generator(x, y, preprocess_generator, batch_size):
    gen = preprocess_generator.flow(x=x, y=y, batch_size=batch_size, shuffle=True)
    while True:
        next_batch = next(gen)
        X = next_batch[0]
        y = next_batch[1]
        y_leaf = y[:, 5:]
        y_high = y[:, :5]
        yield (X, {'leaf_nodes':y_leaf, 'higher_nodes':y_high})
#-----------------------------------------------------------------------------------------------#
# Train Model
print('Beginning training...')
batchsize = 16

history = model.fit_generator(
    batch_generator(
        x=X_train,
        y=y_train,
        preprocess_generator=train_datagen,
        batch_size=batchsize
    ),
    validation_data=(X_val, {'leaf_nodes':y_leaf_val, 'higher_nodes':y_higher_val}),
    epochs=100,
    verbose=1,
    callbacks=callback_list,
    steps_per_epoch=100
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Evaluate on test set
from sklearn.metrics import accuracy_score
print('Evaluating on test set...')
y_test_pred = model.predict(X_test)
with open('y_test_pred', 'wb') as y_writer:
    pickle.dump(y_test_pred, y_writer)

print('Evaluation done and saved!')

# Report
print('Creating report...\n')
y_test_leaf_pred = y_test_pred[0]
y_test_high_pred = y_test_pred[1]

leaf_test_acc = accuracy_score(y_leaf_test, y_test_leaf_pred)
high_test_acc = accuracy_score(y_higher_test, y_test_high_pred)
print('Report:')
print('Higher node acc: {0}'.format(high_test_acc))
print('Leaf node acc: {0}'.format(leaf_test_acc))
#-----------------------------------------------------------------------------------------------#
# Save Model and Training Process
print('\nSaving history...')
with open('history/history_{0}_{1}_gen'.format(model_name, str(args['optimizer'])), 'wb') as file_writer:
    pickle.dump(history.history, file_writer)
print('History saved!\n')