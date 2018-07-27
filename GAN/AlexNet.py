# Construct AlexNet
"""
Network Architecture:
# Contains 60 million parameters
# Activation: RELU

Input takes in images that are 227x227x3

1. Convolution
- Number of filters: 96
- Kernel size: 11x11
- Stride: 4
- Padding: valid
- activation: relu
- input size: 227x227x3

2. Pooling
- Type: Max
- Kernel size: 3x3
- Stride: 2
- Padding: valid

3. Convolution
- Number of filters: 256
- Kernel size: 5x5
- Stride: 1
- Padding: same

4. Pooling
- Type: Max
- Kernel size: 3x3
- Stride: 2
- Padding: valid

5. Convolution
- Number of filters: 384
- Kernel size: 3x3
- Stride: 1
- Padding: same

6. Convolution
- Number of filters: 384
- Kernel size: 3x3
- Stride: 1
- Padding: same

7. Convolution
- Number of filters: 256
- Kernel size: 3x3
- Stride: 1
- Padding: same

8. Pooling
- Type: Max
- Kernel size: 3x3
- Stride: 2
- Padding: valid

# Unroll the last layer to create this fully connected layer
# Number of nodes should be 6*6*256=9216
9. Fully-Connected Layer:
- Number of nodes = 9216

10. Fully-Connected Layer:
- Number of nodes = 4096
- activation: relu

11. Fully-Connected Layer:
- Number of nodes = 4096
- activation: relu

12. Softmax
"""
from keras import Sequential, optimizers, losses
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers.core import Dense
from keras.utils.training_utils import multi_gpu_model
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import cv2


# Construct AlexNet

def AlexNet(num_classes):

    # Initialize model
    model = Sequential()

    # 1. Convolution
    model.add(
        Conv2D(
            filters=96, 
            kernel_size=(11,11),
            strides=(4,4),
            activation='relu',
            padding='valid',
            input_shape=(227, 227, 3)
        )
    )

    # 2. Pooling
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    
    # 3. Convolution
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same"            
        )
    )

    # 4. Pooling
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 5. Convolution
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
    )

    # 6. Convolution
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
    )

    # 7. Convolution
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
    )

    # 8. Pooling
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 9. Flatten to a Fully-Connected Layer
    model.add(Flatten())

    # 10. Fully-Connected Layer
    model.add(
        Dense(
            units=4096,
            activation='relu'
        )
    )

    # 11. Fully-Connected Layer
    model.add(
        Dense(
            units=4096,
            activation='relu'
        )
    )

    # 12. Softmax
    model.add(
        Dense(
            units=num_classes,
            activation='softmax'
        )
    )

    return model


def resize_imgs(imgset):

    list_ = []

    for index, img in enumerate(imgset):

        resized_img = cv2.resize(img, (227, 227))

        list_.append(resized_img)

        if index >= 30000:
            break

    return np.stack(list_)

if __name__ == "__main__":

    # Import dataset
    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype("float")
    testX = testX.astype("float")

    # Resize Training and testing sets
    trainX = resize_imgs(trainX)
    testX = resize_imgs(testX)

    trainY = trainY[:trainX.shape[0]]
    testY = testY[:testY.shape[0]]

    print(trainX.shape)
    print(testX.shape)

    # Normalize data
    mean = np.mean(trainX, axis=0)
    trainX -= mean
    testX -= mean

    # Convert the labels from integers to vectors
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    num_classes = trainY.shape[1]

    print("Number of classes: {0}".format(num_classes))

    model = AlexNet(num_classes=num_classes)
    
    # Compile the model
    adam_optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = multi_gpu_model(model, gpus=2)
    model.compile(
        optimizer=adam_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x=trainX,
        y=trainY,
        batch_size=8,
        verbose=2,
        validation_data=(testX, testY)
    )