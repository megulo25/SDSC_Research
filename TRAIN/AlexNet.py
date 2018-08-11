# Construct AlexNet
"""
Network Architecture:
# Contains 60 million parameters
# Activation: RELU
# Added dropout: 0.4 (happens after pooling)

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
from keras import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.layers.core import Dense

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
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
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
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 5. Convolution
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
    )
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 6. Convolution
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
    )
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

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
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 9. Flatten to a Fully-Connected Layer
    model.add(Flatten())

    # 10. Fully-Connected Layer
    model.add(
        Dense(
            units=4096,
            activation='relu'
        )
    )
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 11. Fully-Connected Layer
    model.add(
        Dense(
            units=4096,
            activation='relu'
        )
    )
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    # 12. Softmax
    model.add(
        Dense(
            units=num_classes,
            activation='softmax'
        )
    )

    return model