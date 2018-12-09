from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import argparse
import imageio
import h5py

model = load_model('models/model.hdf5')
file = h5py.File('data_10.h5')
X = np.array(file['X'])
y = np.array(file['y'])

#-----------------------------------------------------------------------------------------------#
# Get random images
k_samples = 4
idx = random.sample(range(X.shape[0]), 4)
imgs = X[idx]
y_true = y[idx]

y_child_true = y_true[:, 5:]
y_parent_true = y_true[:, :5]
#-----------------------------------------------------------------------------------------------#
# Run Model
y_pred = model.predict(imgs)

y_child = y_pred[0]
y_parent = y_pred[1]
#-----------------------------------------------------------------------------------------------#
# Display results

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

def get_class_name(dict_, y_child, y_parent):
    i=0
    while True:
        child = list(y_child[i])
        parent = list(y_parent[i])
        idx_child = child.index(1) + 5
        idx_parent = parent.index(1)
        i+=1
        yield (dict_[idx_child], dict_[idx_parent])

for i, img in enumerate(imgs):
    # Save image
    imageio.imwrite('imgs/img_{0}.png'.format(i), img)

gen_pred = get_class_name(class_dict, y_child, y_parent)
gen_true = get_class_name(class_dict, y_child_true, y_parent_true)
fig = plt.figure()

# Image 1
y_child_pred_1, y_parent_pred_1 = next(gen_pred)
y_child_true_1, y_parent_true_1 = next(gen_true)

plt.subplot(2, 2, 1)
plt.imshow(mpimg.imread('imgs/img_0.png'))
plt.title('Pred: parent: {0} child: {1}'.format(y_parent_pred_1, y_child_pred_1))
plt.xlabel('True: parent: {0} child: {1}'.format(y_parent_true_1, y_child_true_1))

# Image 2
y_child_pred_2, y_parent_pred_2 = next(gen_pred)
y_child_true_2, y_parent_true_2 = next(gen_true)

plt.subplot(2, 2, 2)
plt.imshow(mpimg.imread('imgs/img_1.png'))
plt.title('Pred: parent: {0} child: {1}'.format(y_parent_pred_2, y_child_pred_2))
plt.xlabel('True: parent: {0} child: {1}'.format(y_parent_true_2, y_child_true_2))

# Image 3
y_child_pred_3, y_parent_pred_3 = next(gen_pred)
y_child_true_3, y_parent_true_3 = next(gen_true)

plt.subplot(2, 2, 3)
plt.imshow(mpimg.imread('imgs/img_2.png'))
plt.title('Pred: parent: {0} child: {1}'.format(y_parent_pred_3, y_child_pred_3))
plt.xlabel('True: parent: {0} child: {1}'.format(y_parent_true_3, y_child_true_3))

# Image 4
y_child_pred_4, y_parent_pred_4 = next(gen_pred)
y_child_true_4, y_parent_true_4 = next(gen_true)

plt.subplot(2, 2, 4)
plt.imshow(mpimg.imread('imgs/img_3.png'))
plt.title('Pred: parent: {0} child: {1}'.format(y_parent_pred_4, y_child_pred_4))
plt.xlabel('True: parent: {0} child: {1}'.format(y_parent_true_4, y_child_true_4))

# Show all plots
plt.show()