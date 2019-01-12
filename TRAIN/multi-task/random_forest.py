import os

import h5py
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
#----------------------------------------------------------------------#
# Import Data and Initialize model
forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=250))
filename = os.path.join(os.getcwd(), 'data_10.h5')

# Load in data
print('Loading in data...')
f = h5py.File(filename)
X = np.array(f['X'])
y = np.array(f['y'])
print('Dataset loaded!\n')

m = X.shape[0]
X = X.flatten().reshape(m, 224*224*3)
#----------------------------------------------------------------------#
# Preprocess
test_size = .3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

del X
del y

X_train = preprocessing.scale(X_train)

# y_train_parent = y_train[:, :5]
# y_train_child = y_train[:, 5:]

# y_test_parent = y_test[:, :5]
# y_test_child = y_test[:, 5:]

# del y_train
# del y_test
#----------------------------------------------------------------------#
# Train (Fit)
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))