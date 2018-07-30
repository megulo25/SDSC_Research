from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from AlexNet import AlexNet

# Import data
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)

# Import AlexNet
model = AlexNet(num_classes=17)

# Output Model Summary
model.summary()

# Make multi-gpu compatible
model = multi_gpu_model(model=model, gpus=2)

# Compile 
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max'
)
callback_list = [checkpoint]

# Train
model.fit(
    x=x,
    y=y,
    batch_size=64,
    epochs=70,
    verbose=2,
    validation_split=0.2,
    shuffle=True,
    callbacks=callback_list
)