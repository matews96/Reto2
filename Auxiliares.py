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
