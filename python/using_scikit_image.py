from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

import numpy as np
import cv2

from PIL import Image
import sys
import os

mydir = os.path.expanduser("~/GitHub/Image-Analysis")

# Read image
image = cv2.imread(mydir + '/photos/test.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

#image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']

titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']

sequence = zip(blobs_list, colors, titles)

for blobs, color, title in sequence:
    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

    #plt.imshow(img, cmap=plt.cm.gray)
    plt.savefig(mydir + '/results/photos/using_scikit_image'+title+'.png')

    #plt.show()
