import cv2
import random
import numpy as np
from DoGdetector import DoGDetector

def main():
    img = cv2.imread('butterfly.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    colorImg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    dogDetector = DoGDetector()
    keyPoints = dogDetector.getKeyPoints(img,1.6,3)
	
	
    #绘制关键点
    for point in keyPoints:
        cv2.circle(colorImg, (round(point[0]),round(point[1])),3,(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    cv2.imshow('ret',colorImg)
    cv2.waitKey(0)

if __name__ =='__main__':
    main()





