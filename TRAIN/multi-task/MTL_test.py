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

y_leaf_true = y_true[:, 5:]
y_high_true = y_true[:, :5]
#-----------------------------------------------------------------------------------------------#
# Run Model
y_pred = model.predict(imgs)

y_leaf = y_pred[0]
y_high = y_pred[1]
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

def get_class_name(dict_, y_leaf, y_high):
    i=0
    while True:
        leaf = list(y_leaf[i])
        high = list(y_high[i])
        idx_leaf = leaf.index(1) + 5
        idx_high = high.index(1)
        i+=1
        yield (dict_[idx_leaf], dict_[idx_high])

for i, img in enumerate(imgs):
    # Save image
    imageio.imwrite('imgs/img_{0}.png'.format(i), img)

gen_pred = get_class_name(class_dict, y_leaf, y_high)
gen_true = get_class_name(class_dict, y_leaf_true, y_high_true)
fig = plt.figure()

# Image 1
y_leaf_pred_1, y_high_pred_1 = next(gen_pred)
y_leaf_true_1, y_high_true_1 = next(gen_true)

plt.subplot(2, 2, 1)
plt.imshow(mpimg.imread('imgs/img_0.png'))
plt.title('Pred: High: {0} Leaf: {1}'.format(y_high_pred_1, y_leaf_pred_1))
plt.xlabel('True: High: {0} Leaf: {1}'.format(y_high_true_1, y_leaf_true_1))

# Image 2
y_leaf_pred_2, y_high_pred_2 = next(gen_pred)
y_leaf_true_2, y_high_true_2 = next(gen_true)

plt.subplot(2, 2, 2)
plt.imshow(mpimg.imread('imgs/img_1.png'))
plt.title('Pred: High: {0} Leaf: {1}'.format(y_high_pred_2, y_leaf_pred_2))
plt.xlabel('True: High: {0} Leaf: {1}'.format(y_high_true_2, y_leaf_true_2))

# Image 3
y_leaf_pred_3, y_high_pred_3 = next(gen_pred)
y_leaf_true_3, y_high_true_3 = next(gen_true)

plt.subplot(2, 2, 3)
plt.imshow(mpimg.imread('imgs/img_2.png'))
plt.title('Pred: High: {0} Leaf: {1}'.format(y_high_pred_3, y_leaf_pred_3))
plt.xlabel('True: High: {0} Leaf: {1}'.format(y_high_true_3, y_leaf_true_3))

# Image 4
y_leaf_pred_4, y_high_pred_4 = next(gen_pred)
y_leaf_true_4, y_high_true_4 = next(gen_true)

plt.subplot(2, 2, 4)
plt.imshow(mpimg.imread('imgs/img_3.png'))
plt.title('Pred: High: {0} Leaf: {1}'.format(y_high_pred_4, y_leaf_pred_4))
plt.xlabel('True: High: {0} Leaf: {1}'.format(y_high_true_4, y_leaf_true_4))

# Show all plots
plt.show()