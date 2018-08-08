# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

#--------------------------------------------------------------------------------------------#
# Import Model:


#--------------------------------------------------------------------------------------------#
# Import Test Data:


#--------------------------------------------------------------------------------------------#
# Run Model on test data:


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