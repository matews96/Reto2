import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Auxiliares as aux

# img = cv2.imread("Data/IMG_0168.JPG")
# img = cv2.resize(img, None, fx=0.15, fy=0.15)
# mask = aux.getMask(img, False)
#cv2.imshow('img', img)

img = []
cap = cv2.VideoCapture(1)

sizexx = 600
sizeyy= 600
sizex = int((960 - sizexx)/2)
sizey = int((1280 - sizeyy)/2)
counter = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = frame[sizex:sizex+sizexx, sizey:sizey+sizeyy]


    cv2.imshow('frame1',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Data/mm.png', frame)
        img = frame
        print("Clustering...")
        break
    if cv2.waitKey(1) & 0xFF == ord('g'):
        counter = counter + 1
        cv2.imwrite('Data/mm'+str(counter)+'.png', frame)

        print(counter)
        img = frame

cv2.destroyAllWindows()
img = cv2.resize(img, None, fx=0.8, fy=0.8)
mask = aux.getMask(img, False)




planes = cv2.split(img)
background = []
for plane in planes:
    background.append(cv2.bitwise_or(cv2.bitwise_and(plane,mask), cv2.bitwise_not(mask)))

img = aux.remBackground(img, True)



clt1, clustered = aux.getClusteredImage(img, show=True)
clt2, clustered = aux.getClusteredImage(clustered, show=True)
clt3, clustered = aux.getClusteredImage(clustered, show=True)

unique, counts = np.unique(clt3.labels_, return_counts=True)
print(dict(zip(unique, counts)))

centers = clt3.cluster_centers_.astype(int)
for i in range(1,8):
    if counts[i] > 1000:
        auxIm = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
        auxIm[:,:,0] = 100
        auxIm[:, :, 1] = centers[i, 0]
        auxIm[:, :, 2] = centers[i, 1]
        auxIm=cv2.cvtColor(auxIm,cv2.COLOR_LAB2BGR)
        cv2.imshow("a"+str(i), auxIm)



plt.scatter(clt3.cluster_centers_[:,0], clt3.cluster_centers_[:,1], s=1)
plt.show()

kernel =  np.matrix('1 1 1 ; 1 1 1; 1 1 1', dtype='uint8')
clustered = cv2.morphologyEx(clustered, cv2.MORPH_OPEN, kernel)
clustered = cv2.morphologyEx(clustered, cv2.MORPH_CLOSE, kernel)
cv2.imshow("gh", clustered)
gray = cv2.cvtColor(clustered,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


planes = cv2.split(img)


# plt.scatter(planes[1].reshape(274670), planes[2].reshape(274670))
# axes = plt.gca()
# axes.set_xlim([0,250])
# axes.set_ylim([0,250])
# plt.show()

imgAB = []
imgAB.append(planes[1])
imgAB.append(planes[2])
imgAB = cv2.merge(imgAB)
img = imgAB




 #represent as row*column,channel number


#plt.axis("off")
#plt.imshow(bar)
#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

