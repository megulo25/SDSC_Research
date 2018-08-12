from keras.utils.training_utils import multi_gpu_model
from helper_functions import split_train_test_dir
from AlexNet import AlexNet
import numpy as np
import os

#-----------------------------------------------------------------------------------------------#
# Split to training and testing set
full_path_to_data = os.path.join(os.getcwd(), 'data', 'nabirds', 'images')
training_percentage = 0.8

message = "Have you already split the images to train and test folders? (y/n): "
resp = input(message)
resp = resp.lower()

if resp == 'n':
    print('Spliting data into training and testing folders...')
    split_train_test_dir(
        dir_of_data=full_path_to_data,
        train_percentage=training_percentage
    )
    print('Split Completed!')

#-----------------------------------------------------------------------------------------------#
# Import AlexNet
class_count = 557
# model = AlexNet(num_classes=class_count)

# Import InceptionNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
model = InceptionResNetV2(weights=None, classes=class_count)


# Output Model Summary
model.summary()

# Make multi-gpu compatible
model = multi_gpu_model(model=model, gpus=2)

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
    directory='data/nabirds/images/train',
    target_size=(224,224),
    batch_size=int(64*2),
    class_mode='categorical'
)

# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = test_datagen.flow_from_directory(
    directory='data/nabirds/images/test',
    target_size=(224,224),
    batch_size=1,
    class_mode='categorical'
)

#-----------------------------------------------------------------------------------------------#
# Optimizer
from keras import optimizers
sgd = optimizers.SGD(lr=0.001, momentum=0.9)

#-----------------------------------------------------------------------------------------------#
# Compile 
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)

#-----------------------------------------------------------------------------------------------#
# Train
print('Training Model!')
import time
t = time.time()
history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    verbose=2
)
print("Training complete!\nTime: {0}secs".format(time.time()-t))

#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
train_acc = history.history['acc']
val_acc = history.history['val_acc']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.save('./history_data/train_acc.npy', train_acc)
np.save('./history_data/val_acc.npy', val_acc)

np.save('./history_data/train_loss.npy', train_loss)
np.save('./history_data/val_loss.npy', val_loss)
#-----------------------------------------------------------------------------------------------#
# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)