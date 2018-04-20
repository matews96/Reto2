import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Auxiliares as aux

img = cv2.imread("Data/IMG_0167.JPG")
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


 #represent as row*column,channel number

axes = plt.gca()
axes.set_xlim([0,250])
axes.set_ylim([0,250])
plt.show()



def getClusteredImage(img2):
    a = img2.shape[0]
    b = img2.shape[1]
    img2 = img.reshape((img.shape[0] * img.shape[1], 2))
    clt = KMeans(n_clusters=5)  # cluster number
    clt.fit(img2)
    clt.fit_predict(img2)

    labels = clt.labels_.reshape(a, b)
    print(len(set(clt.labels_)))

    for i in range(len(set(clt.labels_))):
        planes[1][np.where(labels == i)] = clt.cluster_centers_[i][0]
        planes[2][np.where(labels == i)] = clt.cluster_centers_[i][1]

    final = []
    a = 100 * np.ones(planes[0].shape, dtype='uint8')
    a[np.where(labels == 0)] = 500
    final.append(a)
    final.append(planes[1])
    final.append(planes[2])
    print(planes[0].shape)
    print(a)
    image = cv2.merge(final)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    cv2.imshow('bar', image)


getClusteredImage(img)

#plt.axis("off")
#plt.imshow(bar)
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

