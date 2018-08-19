from keras.utils import multi_gpu_model
from helper_functions import multitask_loss
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#-----------------------------------------------------------------------------------------------#
# Split to training and testing set
full_path_to_data = os.path.join(os.getcwd(), 'data', 'nabirds', 'images')
test_percentage = 0.3

# Import Training Data
X = np.load('X.npy')
y = np.load('y.npy')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)

#-----------------------------------------------------------------------------------------------#
# Import AlexNet
class_count = 557

# Import InceptionNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
model = InceptionResNetV2(weights=None, classes=class_count)

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

train_datagen.fit(X_train)
#-----------------------------------------------------------------------------------------------#
# Optimizer
from keras import optimizers
sgd = optimizers.SGD(lr=0.001, momentum=0.9)
#-----------------------------------------------------------------------------------------------#
# Compile
model.compile(
    loss=multitask_loss,
    optimizer=sgd,
    metrics=['accuracy']
)
#-----------------------------------------------------------------------------------------------#
# Train
model.fit_generator(
    train_datagen.flow(x=X_train, y=y_train, batch_size=32), 
    epochs=100
)