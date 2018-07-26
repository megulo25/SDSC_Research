# set the matplotlib backend so figures can be saved in the background
# (uncomment the lines below if you are using a headless server)
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from minigooglenet import MiniGoogLeNet
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

# Define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):

    maxEpochs = NUM_EPOCHS

    baseLR = INIT_LR

    power = 1.0

    # Compute the new learning rate based on polynomial decay
    alpha = baseLR*(1 - (epoch/float(maxEpochs)))**power

    # Return the new learning rate
    return alpha

# Load the training and testing data, converting the image from 
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

## Normalize input data (look at scikit learn's preprocessing stuff)
# Apply mean subtraction
# Image are 32x32x3
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Construct the image generator for data augmentation and construct
# the set of callbacks.
"""
Data augmentation, in this context, is the process of applying random
transformations to the training images so that the model and generalize
well.
"""
aug = ImageDataGenerator(width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=True,
                        fill_mode="nearest")

callbacks = [LearningRateScheduler(poly_decay)]

# Check to see if we are compiling using just a single GPU
if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = MiniGoogLeNet.build(width=32, height=32, depth=3,
                                classes=10)
else:
    # Otherwise, we are compiling using multiple GPUs
    print("[INFO] training with {0} GPUs...".format(G))

    # We'll store a copy of the model on every GPU and then combine the
    # results from the gradient updates on the CPU
    with tf.device("/cpu:0"):
        """
        We've specified the CPU because the CPU is responsible for
        handling the overhead (such as moving training images on and off 
        GPU memory) while the GPU itself does the heavy lifting.

        In this case the CPU instantiates the base model. We then use keras'
        multi_gpu_model to replicate the model from the CPU to all of our
        GPUs, thereby obtaining single-machine, multi-GPU data parallelism.

        When training our network, images will be batched to each of the GPUs.
        The CPU will obtain the gradients from each GPU and then perform the
        gradient update step.

        We can then compile our model and kick off the training process:
        """

        # Initialize the model
        model = MiniGoogLeNet.build(width=32, height=32, depth=3,
                                    classes=10)

        # Make the model parallel
        model = multi_gpu_model(model, gpus=G)

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
t = time.time()
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=64*G),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX)//(64*G),
    epochs=NUM_EPOCHS,
    callbacks=callbacks, verbose=2)
print("Time took to train: {0} secs".format(time.time()-t))