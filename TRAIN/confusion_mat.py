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
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator =  test_datagen.flow_from_directory(
    directory='data/nabirds/images/test',
    target_size=(227,227),
    batch_size=1,
    class_mode='categorical'
)
#--------------------------------------------------------------------------------------------#
# Run Model on test data:
y_pred = model.predict_generator(test_generator)

# Round off the results to either 0 or 1
for y_p in y_pred:
    

y_test_list = test_generator.filenames

# Split by '/' to get the directory name
y_test = []

for y in y_test_list:
    dir_name = y.split('/')
    y_test.append(dir_name[0])

# One hot encode y_test to compare against y_pred

#--------------------------------------------------------------------------------------------#
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#--------------------------------------------------------------------------------------------#
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()