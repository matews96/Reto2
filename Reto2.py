import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Auxiliares as aux


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    numLabels = numLabels[1:len(np.unique(clt.labels_)) + 1]
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

img = cv2.imread("Data/IMG_0170.JPG")
img = cv2.resize(img, None, fx=0.15, fy=0.15)
mask = aux.getMask(img, True)
cv2.imshow('img', img)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
planes = cv2.split(img)
background = []
for plane in planes:
    background.append(cv2.bitwise_or(cv2.bitwise_and(plane,mask), cv2.bitwise_not(mask)))


img = aux.remBackground(img, True)

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

planes = cv2.split(img)

plt.scatter(planes[1], planes[2])
plt.show()

imgAB = []
imgAB.append(planes[1])
imgAB = cv2.merge(imgAB)
img = imgAB


img = img.reshape((img.shape[0] * img.shape[1],1)) #represent as row*column,channel number
print(img.shape)
clt = KMeans(n_clusters=3) #cluster number
clt.fit(img)

print(clt)

hist = find_histogram(clt)
bar = plot_colors2(hist, clt.cluster_centers_)
bar = cv2.cvtColor(bar, cv2.COLOR_LAB2BGR)

cv2.imshow('bar', bar)


plt.axis("off")
plt.imshow(bar)
plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

