import numpy as np
import cv2

from PIL import Image
import sys
import os

mydir = os.path.expanduser("~/GitHub/Image-Analysis")

# Read image
image = cv2.imread(mydir + '/photos/test.jpg', cv2.IMREAD_COLOR)
image = cv2.resize(image, (0,0), fx=0.2, fy=0.2)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByColor = True
params.blobColor = 255

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector(params)

#im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

# Detect blobs.
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_kp = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
im_kp_small = cv2.resize(im_kp, (0,0), fx=0.2, fy=0.2)
cv2.imwrite(mydir + '/results/photos/test.png',im_kp_small)
