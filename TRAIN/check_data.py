import matplotlib.pyplot as plt
import numpy as np
import h5py

# Load in data
filename = 'data_10.h5'
f = h5py.File(filename, 'r')
X = np.array(f['X'])
y = np.array(f['y'])

# Loop through and see if matches or not.
dict_ = {
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

from scipy import misc
from random import randint
for i in range(X.shape[0]):
    print('---------------------------------------')
    n = randint(0, X.shape[0])
    i=n
    img = X[i]
    p = dict_[y[i][:5].argmax()]
    c = dict_[y[i][5:].argmax()+5]
    print("Parent: {0}".format(p))
    print("Child: {0}".format(c))
    print('Vector: {0}'.format(y[i]))
    print('---------------------------------------')
    img = misc.imshow(img)