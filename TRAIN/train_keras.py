from helper_functions import split_train_test_dir
from keras.utils import multi_gpu_model
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
# Split to training and testing set
message = input('Which nabirds dataset are you working on? (Enter 0 for 10_class or 1 for 555_class): ')
message = int(message)

# Import Training Data
print('Loading dataset...')
if not os.path.isdir('data'):
    os.mkdir('data')

os.chdir('data')
if message == 0:
    message = 'nabirds_10'
    class_count = 10
    if not os.path.isdir('nabirds_10'):
        os.mkdir('nabirds_10')

    if len(os.listdir('nabirds_10')) == 0:
        os.chdir('nabirds_10')
        os.system('wget https://www.dropbox.com/sh/g6aatnar4n5s63g/AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
        os.system('unzip AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
        os.remove('AABBixZUh5SiPvFS7eVVVxlHa?dl=0')
        # Ignore UncroppedCommonGoldeneye
        shutil.rmtree('CommonGoldeneye')
        shutil.rmtree('SpottedTowheee')
        shutil.rmtree('Western Grebe')
        full_path_to_data = os.path.join(os.getcwd(), 'data', message)
        training_dir = os.path.join(full_path_to_data, 'train')
        validation_dir = os.path.join(full_path_to_data, 'test')
        os.chdir('../..')
    else:
        full_path_to_data = os.path.join(os.getcwd(), 'data', message)
        training_dir = os.path.join(full_path_to_data, 'train')
        validation_dir = os.path.join(full_path_to_data, 'test')
        os.chdir('..')
    
elif message == 1:
    class_count = 555
    if not os.path.isdir('nabirds_555'):
        os.mkdir('nabirds_555')

    if len(os.listdir('nabirds_555')) == 0:
        os.chdir('nabirds_555')
        os.system('wget https://www.dropbox.com/s/nf78cbxq6bxpcfc/nabirds.tar.gz')
        os.system('tar xvzf nabirds.tar.gz')
        os.remove('nabirds.tar.gz')
        full_path_to_data = os.path.join(os.getcwd(), 'nabirds', 'images')
        training_dir = os.path.join(full_path_to_data, 'train')
        validation_dir = os.path.join(full_path_to_data, 'test')
        os.chdir('../..')
    else:
        full_path_to_data = os.path.join(os.getcwd(), 'nabirds_555','nabirds', 'images')
        training_dir = os.path.join(full_path_to_data, 'train')
        validation_dir = os.path.join(full_path_to_data, 'test')
        os.chdir('..')
else:
    raise NameError('You need to enter either 0 or 1!')

print('Dataset loaded!\n')

print('Splitting dataset...')
training_percentage = 0.7
# Create Training and Validation folders if they don't exist.
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

if (len(os.listdir(training_dir)) == 0) and (len(os.listdir(validation_dir)) == 0):
    split_train_test_dir(
        dir_of_data=full_path_to_data,
        train_dir=training_dir,
        test_dir=validation_dir,
        train_percentage=training_percentage
    )
print('Dataset split!\n')
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

# Multi-gpu functionality
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

train_generator = train_datagen.flow_from_directory(
    directory=training_dir,
    target_size=(299,299),
    batch_size=16,
    class_mode='categorical'
)
class_indicies = train_generator.class_indices
np.save('class_indicies.npy', class_indicies)
# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1
)

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(299,299),
    batch_size=1,
    class_mode='categorical'
)
#-----------------------------------------------------------------------------------------------#
# Optimizer
from keras import optimizers
# optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callback function (save best model only)
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('./model_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]
#-----------------------------------------------------------------------------------------------#
# Train
print('Beginning training...')
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=500,
    verbose=2,
    callbacks=callback_list
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save the weights
print('Saving weights and architecture...')
model.save_weights('model_weights.h5')
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')

# Training Accuracy and Loss
if not os.path.isdir('history_data'):
    os.mkdir('history_data')
train_acc = history.history['acc']
train_loss = history.history['loss']
np.save('./history_data/train_acc.npy')
np.save('./history_data/train_loss.npy')

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc.npy')
np.save('./history_data/val_loss.npy')

print('History saved!\n')
#-----------------------------------------------------------------------------------------------#
# Save the model architecture
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
print('Everything saved!')