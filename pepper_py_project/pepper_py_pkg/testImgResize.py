

from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


import skimage.transform

ha = np.array([0,1,2,3,4,5,6,7,8,9],dtype='float64')

#print(skimage.transform.downscale_local_mean( ha, (2) ) )

hi = np.array([    [0.0,0.0,0.1,0.1],
                   [0.0,0.0,0.1,0.1],
                   [0.2,0.2,0.3,0.3],
                   [0.2,0.2,0.3,0.3]],dtype='float64')


print('downscale')
print(skimage.transform.downscale_local_mean( ha, (2,2) ) )

        
#print(hi)
#plt.imshow(hi)
plt.imshow(skimage.transform.rescale(hi,3))


