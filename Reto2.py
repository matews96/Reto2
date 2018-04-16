import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('Data/IMG_0169.JPG')
imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

a = img.reshape((img.shape[0] * img.shape[1], 3))

kmeans = KMeans(n_clusters=4, random_state=0).fit(a)



def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist




def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


hist = centroid_histogram(kmeans)
bar = plot_colors(hist, kmeans.cluster_centers_)

plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()

