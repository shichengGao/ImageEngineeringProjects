import math
import numpy as np
#输入图像需满足numpy.ndarray格式

class GaussianConv:
    def __init__(self,sigma):
        self.size = 5
        #get Gaussian kernel
        #根据高斯核线性可分的特点，把二维高斯核拆分成两个一维高斯核
        #而且两个一维核的卷积可以同时执行
        self.rowkernel = np.array([self.__getGaussianValue(0,i,sigma) for i in range(-2,3)])
        sum = np.sum(self.rowkernel)
        self.rowkernel /= sum

    def  __getGaussianValue(self,u,v,sigma):
        return (1/(2 * math.pi * sigma**2)) * math.e**(-(u**2+v**2) / 2*sigma**2)

    def conv(self,img:np.ndarray) -> np.ndarray:
        if len(img.shape)>2 and img.shape[-1] > 1:
            raise Exception("channels of image is greater than 1.")
        m,n = img.shape
        ret = np.zeros((m,n),dtype=np.uint8)

	#两个一维高斯卷积
        for row in range(m):
            for col in range(n):
                sum = 0
                for i in range(-2,3):
                    if 0<=col+i<n:
                        sum += self.rowkernel[i+2] * img[row][col+i]
                ret[row][col] = sum

        for row in range(m):
            for col in range(n):
                sum = 0
                for i in range(-2,3):
                    if 0<=row+i<m:
                        sum += self.rowkernel[i+2] * ret[row+i][col]
                ret[row][col] = sum
        return ret
        
