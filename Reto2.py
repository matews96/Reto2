import numpy as np
import cv2
import Auxiliares
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread('Data/IMG_0173.JPG')
img = cv2.resize(img, None, fx=0.15, fy=0.15)
imgblurred = cv2.GaussianBlur(img, (3, 3),0)
img = cv2.addWeighted(img,1.5,imgblurred,-0.5,0)
imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
imgY = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
imHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
Auxiliares.printComponents(imHSV)
planes = cv2.split(imHSV)
imgBW = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(planes[1],90,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(planes[1],80,255,cv2.THRESH_TRUNC)
ret, thresh2 = cv2.threshold(thresh2,50,255,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# thresh = cv2.dilate(thresh2,kernel,iterations = 1)

im2, contours, hierarchy = cv2.findContours(thresh2, 1, 2)
# moments = [cv2.moments(cnt) for cnt in contours]
# centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
# for c in centroids:
#     cv2.circle(img, c, 5, (0, 0, 0))

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.imshow('window', thresh2)

cv2.namedWindow('centers', cv2.WINDOW_NORMAL)
cv2.imshow('centers', img)






#contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img, contours, -1, (0,255,0), 3)

# plt.figure()
# plt.axis("off")
# plt.imshow(img)
# plt.show()
# plt.imshow(imgBW, cmap='gray')
# plt.show()
# plt.imshow(imgLAB)
# plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

