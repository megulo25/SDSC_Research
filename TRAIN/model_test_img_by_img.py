from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Load model
model = load_model('model_best.hdf5')

validation_dir = os.path.join(os.getcwd(), 'data', 'nabirds_9', 'test')

# Generator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(299,299),
    batch_size=1,
    class_mode='categorical'
)

data_base_dir = os.path.join(os.getcwd(), 'data', 'nabirds_9')
filename = validation_generator.filenames
class_indicies = validation_generator.class_indices
class_list = list(class_indicies.keys())
y_true = validation_generator.classes


y_pred = model.predict_generator(validation_generator)
np.save('y_pred.npy', y_pred)

for i in range(len(y_true)):
    img_old = mpimg.imread(os.path.join(data_base_dir,filename[i]))
    img = cv2.resize(img_old, (299,299))
    img = np.reshape(img, (1,299,299,3))
    y_pred_ex = class_list[y_pred[i].argmax()]
    y_true_ex = class_list[y_true[i]]
    plt.figure()
    plt.imshow(img_old)
    plt.title('True Class: {0},  Predicted Class: {1}'.format(y_true_ex, y_pred_ex))
    plt.show()