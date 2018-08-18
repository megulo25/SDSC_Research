# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

#--------------------------------------------------------------------------------------------#
# Import Model:
from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')

#--------------------------------------------------------------------------------------------#
# Prepare Test Data:
from helper_functions import get_val_images
val_directory = 'data/nabirds/images/test'
class_list, img_list, dict_ = get_val_images(val_directory)
#--------------------------------------------------------------------------------------------#
# Run Model on test data:

from helper_functions import predict_class

y_pred = []
y_true = []
for class_, img in zip(class_list,img_list):
    try:
        y_p = predict_class(model, img)
        y_pred.append(dict_[y_p])
        y_true.append(class_)
    except:
        print(img.shape)


#--------------------------------------------------------------------------------------------#
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)

#--------------------------------------------------------------------------------------------#
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_true,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=y_true, normalize=True,
                      title='Normalized confusion matrix')

plt.show()