from helper_functions import multitask_loss, random_images, resize_and_reshape_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
from keras.models import load_model
import os

# Load Model
model = load_model('model_multi_task_best.hdf5', custom_objects={'multitask_loss':multitask_loss})

# Get a list of images to test against
n = 10
dir_ = os.path.join(os.getcwd(), 'data', 'nabirds_10')
list_ = random_images(n, dir_)

# Class dictionary
class_dict = {
    0: 'Goldeneye',
    1: 'Grosbeak_Bunting',
    2: 'Towhee',
    3: 'Grebe',
    4: 'Scaup_Duck',
    5: 'Barrows_Goldeneye',
    6: 'Blue_Grosbeak',
    7: 'Clarks_Grebe',
    8: 'Common_Goldeneye',
    9: 'Eastern_Towhee',
    10: 'Indigo_Bunting',
    11: 'Lesser_Scaup',
    12: 'Ring_Necked_Duck',
    13: 'Spotted_Towhee',
    14: 'Western_Grebe'
}

# Directory for Plots
if not os.path.isdir('multi-task-test'):
    os.mkdir('multi-task-test')

# Loop through random images and display results
for img_name in list_:

    img = mpimg.imread(img_name)
    img_old = img
    
    # Resize and reshape
    img = resize_and_reshape_image(img)
    y_true = img_name.split('/')[-2]

    # Make prediction
    y = model.predict(img)
    y_p = y[0][:5].argmax()
    y_c = y[0][5:].argmax() + 5

    parent = class_dict[y_p]
    child = class_dict[y_c]

    # Display result
    plt.figure()
    plt.imshow(img_old)
    plt.title('True Class: {0}, Parent Class: {1}, Child Class: {2}'.format(y_true, parent, child))
    plt.savefig('multi-task-test/{0}_{1}.jpg'.format(y_true, randint(1, 1e9)))