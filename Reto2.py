import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('Data/IMG_0169.JPG')
imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

a = img[:,:,1]

kmeans = KMeans(n_clusters=3, random_state=0).fit(a)

imgL = imgLAB[:,:,0]
imgA = imgLAB[:,:,1]
imgB = imgLAB[:,:,2]


plt.imshow(imgL)
plt.show()
plt.imshow(imgA)
plt.show()
plt.imshow(imgB)

plt.show()