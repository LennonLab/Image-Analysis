from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
from PIL import Image
import sys
import os


""" This is simply a script I (ken) made to show how to convert an image to
    RGB, to record the original image's shape, then convert the image to a 1D
    array (which is what some programs like to operate on), and then convert
    it back to an RGB. """

mydir = os.path.expanduser("~/GitHub/Image-Analysis")

# Read image
X = Image.open(mydir + '/photos/test.jpg').convert('RGBA')
#X.show()

#img = Image.open('orig.png').convert('RGBA')
arr = np.array(X)

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr2, 'RGBA')
img2.show()
