import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Auxiliares as aux

img = cv2.imread("Data/IMG_0173.JPG")
img = cv2.resize(img, None, fx=0.15, fy=0.15)
mask = aux.getMask(img, True)
cv2.imshow('img', img)

planes = cv2.split(img)
background = []
for plane in planes:
    background.append(cv2.bitwise_or(cv2.bitwise_and(plane,mask), cv2.bitwise_not(mask)))

img = aux.remBackground(img, True)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
planes = cv2.split(img)


plt.scatter(planes[1].reshape(274670), planes[2].reshape(274670))
axes = plt.gca()
axes.set_xlim([0,250])
axes.set_ylim([0,250])
plt.show()

imgAB = []
imgAB.append(planes[1])
imgAB.append(planes[2])
imgAB = cv2.merge(imgAB)
img = imgAB


img2 = img.reshape((img.shape[0] * img.shape[1],2)) #represent as row*column,channel number
print(img.shape)
axes = plt.gca()
axes.set_xlim([0,250])
axes.set_ylim([0,250])
plt.show()


clt = KMeans(n_clusters=7) #cluster number
clt.fit(img2)
clt.fit_predict(img2)

labels = clt.labels_.reshape(img.shape[0], img.shape[1])
planes[1][np.where(labels==0)] = clt.cluster_centers_[0][0]
planes[1][np.where(labels==1)] = clt.cluster_centers_[1][0]
planes[1][np.where(labels==2)] = clt.cluster_centers_[2][0]
planes[1][np.where(labels==3)] = clt.cluster_centers_[3][0]
planes[1][np.where(labels==4)] = clt.cluster_centers_[4][0]
planes[1][np.where(labels==5)] = clt.cluster_centers_[5][0]
planes[1][np.where(labels==6)] = clt.cluster_centers_[6][0]


planes[1][np.where(labels==0)] = clt.cluster_centers_[0][1]
planes[1][np.where(labels==1)] = clt.cluster_centers_[1][1]
planes[1][np.where(labels==2)] = clt.cluster_centers_[2][1]
planes[1][np.where(labels==3)] = clt.cluster_centers_[3][1]
planes[1][np.where(labels==4)] = clt.cluster_centers_[4][1]
planes[1][np.where(labels==5)] = clt.cluster_centers_[5][1]
planes[1][np.where(labels==6)] = clt.cluster_centers_[6][1]


final = []
final.append(planes[0])
final.append(planes[1])
final.append(planes[2])

image = cv2.merge(final)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

#hist = find_histogram(clt)
#bar = plot_colors2(hist, clt.cluster_centers_)
#bar = cv2.cvtColor(bar, cv2.COLOR_LAB2BGR)

cv2.imshow('bar', image)


#plt.axis("off")
#plt.imshow(bar)
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

