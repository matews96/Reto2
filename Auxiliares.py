import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation

def printComponents(image):
    planes = cv2.split(image)
    cv2.namedWindow('Componente 1', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 1', planes[0])

    cv2.namedWindow('Componente 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 2', planes[1])

    cv2.namedWindow('Componente 3', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 3', planes[2])


def printComponentss(image):
    planes = cv2.split(image)
    cv2.namedWindow('Componente 11', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 11', planes[0])

    cv2.namedWindow('Componente 12', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 12', planes[1])

    cv2.namedWindow('Componente 13', cv2.WINDOW_NORMAL)
    cv2.imshow('Componente 13', planes[2])

def normalizeBGR(image):
    planes = cv2.split(image)
    np.seterr(divide='ignore', invalid='ignore')
    out =[]
    constante = (planes[0]+planes[1]+planes[2])/255
    out.append(planes[0])
    out.append(planes[1])
    out.append(planes[2])
    out = cv2.merge(out)
    return out

def getMask(img, show=False):

    imgblurred = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.5, imgblurred, -0.5, 0)
    imHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    planes = cv2.split(imHSV)
    ret, thresh2 = cv2.threshold(planes[1], 80, 255, cv2.THRESH_TRUNC)
    ret, thresh2 = cv2.threshold(thresh2, 50, 255, cv2.THRESH_BINARY)

    if show:
        cv2.imshow('mask', thresh2)

    return thresh2

def remBackground(img, show=False):
    mask = getMask(img, False)
    planes = cv2.split(img)
    background = []
    for plane in planes:
        background.append(cv2.bitwise_or(cv2.bitwise_and(plane, mask), cv2.bitwise_not(mask)))

    img = cv2.merge(background)

    if show:
        cv2.imshow('Background removed', img)

    return img

def getClusteredImage(img, clusters=9, show=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a = img.shape[0]
    b = img.shape[1]
    planes = cv2.split(img)
    imgAB = []
    imgAB.append(planes[1])
    imgAB.append(planes[2])
    imgAB.append(planes[0])
    imgAB = cv2.merge(imgAB)
    img = imgAB
    img2 = img.reshape((a*b, 3))

    clt = KMeans(n_clusters=clusters)  # cluster number
    clt.fit(img2)
    clt.fit_predict(img2)

    labels = clt.labels_.reshape(a, b)
    print(len(set(clt.labels_)))

    for i in range(len(set(clt.labels_))):
        planes[1][np.where(labels == i)] = clt.cluster_centers_[i][0]
        planes[2][np.where(labels == i)] = clt.cluster_centers_[i][1]
        planes[0][np.where(labels == i)] = clt.cluster_centers_[i][2]

    final = []
    a = 100 * np.ones(planes[0].shape, dtype='uint8')
    a[np.where(labels == 0)] = 500
    final.append(a)
    final.append(planes[1])
    final.append(planes[2])
    image = cv2.merge(final)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    if show:
        cv2.imshow('colors clusters', image)

    return clt, image



