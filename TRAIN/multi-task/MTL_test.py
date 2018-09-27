from keras.models import load_model
from random import choice
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import cv2
import os

# Arg parser
arg = argparse.ArgumentParser()
arg.add_argument('-m', '--model', required=True, help='Path to the trained model')
args = vars(arg.parse_args())

# Import Hierarchy dict and Class dict
a = os.getcwd()
b = a.split('/')
b = b[:-1]
c = '/'.join(b)

hierarchy_dict = np.load(os.path.join(c, 'data', 'hierarchy_dict.npy')).item()
class_dict = np.load(os.path.join(c, 'data', 'class_dict_555.npy')).item()

# Import model
model = load_model(args['model'])

# Get a list of sub-directories
path_to_subdirectories = os.path.join(c, 'data', 'nabirds_555', 'nabirds', 'images')
for (_, sub_directory_list, _) in os.walk(path_to_subdirectories):
    True
    break

# Randomly select images to test model
for n in range(100):
    item = choice(sub_directory_list)
    path_to_images = os.path.join(path_to_subdirectories, item)
    for (_, _, img_names) in os.walk(path_to_images):
        True
        pass

    # Get random image
    img_name = choice(img_names)
    full_path = os.path.join(path_to_images, img_name)

    # Pre-process image
    img_original = mpimg.imread(full_path)
    img = cv2.resize(img_original, (299,299))
    img = img.astype('float') / 255.0
    img = np.reshape(img, (1,299,299,3))

    # Get ground truth class names
    hierarchy = list(hierarchy_dict[int(item)])
    hierarchy.sort()

    ground_truth_classes = []
    for j in hierarchy:
        ground_truth_classes.append(class_dict[j])

    # Model prediction (grab top 5 results)
    predicted_classes = []
    p = model.predict(img)[0]
    idxs = np.argsort(p)[::-1][:5]
    idxs.sort()

    for k in idxs:
        predicted_classes.append(class_dict[k])

    # Plot model predictions
    n = n % 4
    n = [1,2,3,4][n]
    plt.subplot(2,2,n)
    plt.imshow(img_original)
    plt.title('Ground Truth: {0}'.format(ground_truth_classes))
    plt.xlabel('Prediction: {0}'.format(predicted_classes))

    if n % 4 == 0:
        plt.show()