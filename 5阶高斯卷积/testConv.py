import cv2
import numpy as np
import argparse
from GaussianConv import GaussianConv

#这只是个测试程序
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','-input',type=str,help='set input image',default='lena.jpg')
    parser.add_argument('-width',type=int,default=180)
    parser.add_argument('-height',type=int,default=180)
    args = parser.parse_args()

    filename = args.i
    width = args.width
    height = args.height

    img = cv2.imread(filename)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(height,width))
    cv2.imwrite('ori.jpg',img)
    
    gconv = GaussianConv(sigma=2.0)
    ret = gconv.conv(img)
    cv2.namedWindow("result image",cv2.WINDOW_NORMAL)
    cv2.imshow("result image",ret)
    cv2.waitKey(0)
    cv2.imwrite('ret.jpg',ret)
    
    

if __name__ =='__main__':
    main()
    


