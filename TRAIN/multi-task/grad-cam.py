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

# Get Leaf and high outputs
y_leaf_idx = np.argmax(y_pred[0])
y_high_idx = np.argmax(y_pred[1])

y_leaf_output = model.output[0][:, y_leaf_idx]
y_high_output = model.output[1][:, y_high_idx]
#----------------------------------------------------------------------#
# Apply heatmap

last_conv_layer = model.get_layer("block5_conv3")

leaf_grads = K.gradients(y_leaf_output, last_conv_layer.output)[0]
high_grads = K.gradients(y_high_output, last_conv_layer.output)[0]

leaf_pool = K.mean(leaf_grads, axis=(0, 1, 2))
high_pool = K.mean(high_grads, axis=(0, 1, 2))

leaf_iterate = K.function([model.input], [leaf_pool, last_conv_layer.output[0]])
high_iterate = K.function([model.input], [high_pool, last_conv_layer.output[0]])

leaf_pool_value, leaf_conv_layer_value = leaf_iterate([x])
high_pool_value, high_conv_layer_value = high_iterate([x])

for i in range(512):
    leaf_conv_layer_value[:, :, i] *= leaf_pool_value[i]
    high_conv_layer_value[:, :, i] *= high_pool_value[i]

leaf_heatmap = np.mean(leaf_conv_layer_value, axis=-1)
leaf_heatmap = np.maximum(leaf_heatmap, 0)
leaf_heatmap /= np.max(leaf_heatmap)

high_heatmap = np.mean(high_conv_layer_value, axis=-1)
high_heatmap = np.maximum(high_heatmap, 0)
high_heatmap /= np.max(high_heatmap)

#----------------------------------------------------------------------#
# Plot result
img_old = mpimg.imread(img_path)
img_old = cv2.resize(img_old, (224, 224))


# leaf_heatmap = cv2.resize(leaf_heatmap, (img_old.shape[1], img_old.shape[0]))
# leaf_heatmap = np.uint8(255 * leaf_heatmap)
# leaf_heatmap = cv2.applyColorMap(leaf_heatmap, cv2.COLORMAP_JET)
# leaf_superimposed_img = cv2.addWeighted(img_old, 0.6, leaf_heatmap, 0.4, 0)

plt.subplot(1, 2, 1)
plt.imshow(img_old)

plt.subplot(1, 2, 2)
plt.imshow(high_superimposed_img)
plt.show()