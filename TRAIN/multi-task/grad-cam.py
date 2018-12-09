# Modules
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
import argparse
import cv2
#----------------------------------------------------------------------#
# Argparser
arg = argparse.ArgumentParser()
arg.add_argument('-img', '--IMG', required=True, help='Path to img')
arg.add_argument('-model', '--MODEL', required=True, help='Path to model')
args = vars(arg.parse_args())
#----------------------------------------------------------------------#
# Import Image
img_path = str(args['IMG'])
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#----------------------------------------------------------------------#
# Import Model
model = load_model(str(args['MODEL']))
#----------------------------------------------------------------------#
# Run Model
y_pred = model.predict(x)
y_child = y_pred[0]
y_parent = y_pred[1]

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
        child = list(y_child[i].round())
        parent = list(y_parent[i].round())
        idx_child = child.index(1) + 5
        idx_parent = parent.index(1)
        i+=1
        yield (dict_[idx_child], dict_[idx_parent])

gen = get_class_name(class_dict, y_child, y_parent)

y_child_label, y_parent_label = next(gen)

# Get child and parent outputs
y_child_idx = np.argmax(y_pred[0])

y_child_output = model.output[0][:, y_child_idx]
#----------------------------------------------------------------------#
# Apply heatmap

last_conv_layer = model.get_layer("block5_conv3")

child_grads = K.gradients(y_child_output, last_conv_layer.output)[0]

child_pool = K.mean(child_grads, axis=(0, 1, 2))

child_iterate = K.function([model.input], [child_pool, last_conv_layer.output[0]])

child_pool_value, child_conv_layer_value = child_iterate([x])

for i in range(512):
    child_conv_layer_value[:, :, i] *= child_pool_value[i]

child_heatmap = np.mean(child_conv_layer_value, axis=-1)
child_heatmap = np.maximum(child_heatmap, 0)
child_heatmap /= np.max(child_heatmap)
#----------------------------------------------------------------------#
# Plot result
img_old = mpimg.imread(img_path)
img_old = cv2.resize(img_old, (224, 224))



child_heatmap = cv2.resize(child_heatmap, (img_old.shape[1], img_old.shape[0]))
child_heatmap = np.uint8(255 * child_heatmap)
child_heatmap = cv2.applyColorMap(child_heatmap, cv2.COLORMAP_JET)
child_superimposed_img = cv2.addWeighted(img_old, 0.6, child_heatmap, 0.4, 0)

plt.subplot(1, 2, 1)
plt.imshow(img_old)
plt.title('True:\nParent: {0}\nChild: {1}'.format(str('Grebe'), str('Clarks Grebe')))


plt.subplot(1, 2, 2)
plt.imshow(child_superimposed_img)
plt.title('Pred:\nParent: {0}\nChild: {1}'.format(str(y_parent_label), str(y_child_label)))
plt.show()