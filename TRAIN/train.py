from keras.utils.training_utils import multi_gpu_model
from helper_functions import split_train_test_dir
from AlexNet import AlexNet
import numpy as np
import os

#-----------------------------------------------------------------------------------------------#
# Split to training and testing set
full_path_to_data = os.path.join(os.getcwd(), 'data', 'nabirds', 'images')
training_percentage = 0.99

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
model = AlexNet(num_classes=class_count)

# Output Model Summary
model.summary()


# Make multi-gpu compatible
# model = multi_gpu_model(model=model, gpus=2)

#-----------------------------------------------------------------------------------------------#
# Image Pre-processing
from keras.preprocessing.image import ImageDataGenerator

# Training Generator
train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode="nearest")


train_generator = train_datagen.flow_from_directory(
    directory='data/nabirds/images/train',
    target_size=(227,227),
    batch_size=1,
    class_mode='categorical'
)

# Validation Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = test_datagen.flow_from_directory(
    directory='data/nabirds/images/test',
    target_size=(227,227),
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
    epochs=1,
    verbose=2
)
np.save('history.npy', history)
print("Training complete!\nTime: {0}secs".format(time.time()-t))

#-----------------------------------------------------------------------------------------------#
# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)