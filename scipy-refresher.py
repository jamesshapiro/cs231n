#!/usr/bin/env python3 -tt

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

#========== Tint and Reshape Cat ==========
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)
# uint8 (400, 248, 3)

img_tinted = img * [1.0, 0.66, 0.33]
img_tinted = imresize(img_tinted, (400, 200))
imsave('assets/cat_tinted.jpg', img_tinted)

#========== compute distance between all pairs of points ==========
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

d = squareform(pdist(x, 'euclidean'))
print(d)
'''[[ 0.          1.41421356  2.23606798]
 [ 1.41421356  0.          1.        ]
 [ 2.23606798  1.          0.        ]]'''

d = squareform(pdist(x, 'hamming'))
print(d)
'''[[ 0.   1.   1. ]
 [ 1.   0.   0.5]
 [ 1.   0.5  0. ]]'''

d = squareform(pdist(x, 'cityblock'))
print(d)
'''
[[ 0.  2.  3.]
 [ 2.  0.  1.]
 [ 3.  1.  0.]]
'''
