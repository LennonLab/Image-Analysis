from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

from PIL import Image
import sys
import os


mydir = os.path.expanduser("~/GitHub/Image-Analysis")

# Read image
img = Image.open(mydir + '/photos/test.jpg').convert('RGBA')

arr = np.array(img)
X = arr.ravel()

fig = plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.savefig(mydir + '/results/photos/image_as_analyzable_object.png')


# Compute clustering with Birch
birch_models = [Birch(threshold=1.7, n_clusters=None)]

#for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
t1 = time()
birch_model.fit(X)
t2 = time()
print("Birch %s as the final step took %0.2f seconds" % (info, (t2))

# Plot result
labels = birch_model.labels_
centroids = birch_model.subcluster_centers_
n_clusters = np.unique(labels).size
print("n_clusters : %d" % n_clusters)

ax = fig.add_subplot(1, 1, 1)
for this_centroid, k, col in zip(centroids, range(n_clusters), colors):
    mask = labels == k
    ax.plot(X[mask, 0], X[mask, 1], 'w',
            markerfacecolor=col, marker='.')
    if birch_model.n_clusters is None:
        ax.plot(this_centroid[0], this_centroid[1], '+', markerfacecolor=col,
                markeredgecolor='k', markersize=5)

ax.set_ylim([-25, 25])
ax.set_xlim([-25, 25])
ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)

#cv2.imwrite(mydir + '/results/photos/using_scikit_learn_clustering.png', fig)
