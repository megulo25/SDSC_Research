from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Load model
model = load_model('model_best_nabirds_10.hdf5')

validation_dir = os.path.join(os.getcwd(), 'data', 'nabirds_10', 'test')

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

data_base_dir = os.path.join(os.getcwd(), 'data', 'nabirds_10')
filename = validation_generator.filenames
class_indicies = validation_generator.class_indices
class_list = ['Barrows_Goldeneye', 'Blue_grosbeak', 'Clarks_Grebe', 'Common_Goldeneye', 'Eastern_towhee', 'Indigo_Bunting', 'Lesser_scaup', 'Ring_necked_duck', 'Spotted_Towheee', 'Western_Grebe']
y_true = validation_generator.classes


def adj_img(img_old):
    img = cv2.resize(img_old, (299,299))
    img = np.reshape(img, (1,299,299,3))
    return img


from random import randint
for i in range(len(y_true)):
    rand_num = randint(1, len(y_true))
    file_path = os.path.join(data_base_dir,filename[rand_num])
    print('filepath: {0}'.format(file_path))
    img_old = mpimg.imread(file_path)
    img = adj_img(img_old)
    y_pred_ex = model.predict(img)
    y_pred_ex = y_pred_ex.round().argmax()
    y_pred_ex = class_list[y_pred_ex]
    y_true_ex = class_list[y_true[rand_num]]
    plt.figure()
    plt.imshow(img_old)
    plt.title('True Class: {0},  Predicted Class: {1}'.format(y_true_ex, y_pred_ex))
    plt.show()