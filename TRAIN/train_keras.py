from keras.utils import multi_gpu_model
from helper_functions import multitask_loss
import numpy as np
import os
import sys

gpu_number = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"]="{0}".format(gpu_number)
#-----------------------------------------------------------------------------------------------#
# Split to training and testing set
full_path_to_data = os.path.join(os.getcwd(), 'data', 'nabirds', 'images')
test_percentage = 0.3

# Import Training Data
print('Loading dataset...')
X = np.load('X.npy')
y = np.load('y.npy')
print('Dataset loaded!\n')

print('Splitting dataset...')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage)
print('Dataset split!\n')
#-----------------------------------------------------------------------------------------------#
# Import AlexNet
class_count = 1011

# Import InceptionNet
print('Loading in model...')
from keras.applications.inception_resnet_v2 import InceptionResNetV2
model = InceptionResNetV2(weights=None, classes=class_count)
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

train_datagen.fit(X_train)
#-----------------------------------------------------------------------------------------------#
# Optimizer
from keras import optimizers
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#-----------------------------------------------------------------------------------------------#
# Compile
from keras import metrics
model.compile(
    loss=multitask_loss,
    optimizer=adam,
    metrics=['accuracy', metrics.sparse_categorical_accuracy]
)
#-----------------------------------------------------------------------------------------------#
# Train
print('Beginning training...')
history = model.fit_generator(
    train_datagen.flow(x=X_train, y=y_train, batch_size=32*4), 
    epochs=250
)
print('Training complete!\n')
#-----------------------------------------------------------------------------------------------#
# Save the weights
print('Saving weights and architecture...')
model.save_weights('model_weights_{0}.h5'.format(gpu_number))
#-----------------------------------------------------------------------------------------------#
# Save training accuracy and testing accuracy:
print('Saving history...')
train_acc = history.history['acc']
val_acc = history.history['val_acc']

train_loss = history.history['loss']
val_loss = history.history['val_loss']

np.save('./history_data/train_acc_{0}.npy'.format(gpu_number), train_acc)
np.save('./history_data/val_acc_{0}.npy'.format(gpu_number), val_acc)

np.save('./history_data/train_loss_{0}.npy'.format(gpu_number), train_loss)
np.save('./history_data/val_loss_{0}.npy'.format(gpu_number), val_loss)
print('History saved!\n')
#-----------------------------------------------------------------------------------------------#
# Save the model architecture
model_json = model.to_json()
with open('model_architecture_{0}.json'.format(gpu_number), 'w') as json_file:
    json_file.write(model_json)
print('Everything saved!')