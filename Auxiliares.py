import numpy as np
import cv2

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

def getMask(img):

    imgblurred = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.addWeighted(img, 1.5, imgblurred, -0.5, 0)
    imHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    planes = cv2.split(imHSV)
    ret, thresh2 = cv2.threshold(planes[1], 80, 255, cv2.THRESH_TRUNC)
    ret, thresh2 = cv2.threshold(thresh2, 50, 255, cv2.THRESH_BINARY)
    return thresh2
