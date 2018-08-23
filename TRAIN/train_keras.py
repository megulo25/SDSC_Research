from helper_functions import split_train_test_dir
import numpy as np
import os
import sys
import shutil

gpu_number = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(gpu_number)
#-----------------------------------------------------------------------------------------------#
# Split to training and testing set

# Import Training Data
print('Loading dataset...')
if not os.path.isdir('data'):
    os.mkdir('data')
if len(os.listdir('data')) == 0:
    os.chdir('data')
    os.system('wget https://www.dropbox.com/sh/g6aatnar4n5s63g/AABBixZUh5SiPvFS7eVVVxlHa')
    os.system('unzip AABBixZUh5SiPvFS7eVVVxlHa')
    os.remove('AABBixZUh5SiPvFS7eVVVxlHa')

    # Ignore UncroppedCommonGoldeneye
    shutil.rmtree('UncroppedCommonGoldeneye')
    os.chdir('..')
print('Dataset loaded!\n')

print('Splitting dataset...')

full_path_to_data = os.path.join(os.getcwd(), 'data')
training_dir = os.path.join(full_path_to_data, 'train')
validation_dir = os.path.join(full_path_to_data, 'test')
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
class_count = 9

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
                                    rescale=1./255,
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

# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
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
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss='mean_squared_error',
    optimizer=adam,
    metrics=['accuracy']
)
#-----------------------------------------------------------------------------------------------#
# Train
print('Beginning training...')
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=200,
    verbose=2
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save the weights
print('Saving weights and architecture...')
model.save_weights('model_weights_{0}.h5'.format(gpu_number))
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')

# Training Accuracy and Loss
if not os.path.isdir('history_data'):
    os.mkdir('history_data')
train_acc = history.history['acc']
train_loss = history.history['loss']
np.save('./history_data/train_acc_{0}.npy'.format(gpu_number), train_acc)
np.save('./history_data/train_loss_{0}.npy'.format(gpu_number), train_loss)

# Validation Accuracy and Loss
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
np.save('./history_data/val_acc_{0}.npy'.format(gpu_number), val_acc)
np.save('./history_data/val_loss_{0}.npy'.format(gpu_number), val_loss)

print('History saved!\n')
#-----------------------------------------------------------------------------------------------#
# Save the model architecture
model_json = model.to_json()
with open('model_architecture_{0}.json'.format(gpu_number), 'w') as json_file:
    json_file.write(model_json)
print('Everything saved!')