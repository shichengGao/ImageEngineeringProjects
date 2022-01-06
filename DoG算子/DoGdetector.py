import math

import numpy as np
import cv2

class DoGDetector:
    def __init__(self):
        self.gaussian_pyr = list()
        self.dog_pyr = list()

        self.__SIFT_IMAGE_BORDER = 5
        self.__MAX_INTERP_ITER_STEPS = 5
        self.__CURV_THRESHOLD = 10

    def getKeyPoints(self,img: np.ndarray, sigma: float, nIntervals: int,contr_thr = 0.04):
        if img.dtype != np.float:
            img = img.astype(float)/255
        nOctaves = int(math.log2(min(img.shape)) - 2)


        #build gaussian pyramid
        self.__buildGaussianPyramid(img, nOctaves, nIntervals, sigma)
        #build DoG pyramid
        self.__buildDoGPyramid(nOctaves,nIntervals)

        keyPoints = []
        prelim_contr_thr = 0.5 * contr_thr / nIntervals
        for i_octave in range(nOctaves):
           for i_interval in range(nIntervals):
               for row in range(self.__SIFT_IMAGE_BORDER,self.dog_pyr[i_octave][i_interval].shape[0]-self.__SIFT_IMAGE_BORDER):
                   for col in range(self.__SIFT_IMAGE_BORDER,self.dog_pyr[i_octave][i_interval].shape[1]-self.__SIFT_IMAGE_BORDER):
                       if math.fabs(self.dog_pyr[i_octave][i_interval][row][col] > contr_thr):
                           if self.__isExtrema(i_octave,i_interval,row,col):
                               extremum = self.__getInterpolatedExtremum(i_octave,i_interval,row,col,prelim_contr_thr)
                               if extremum:
                                   i, r, c, xcor ,ycor = extremum[0],extremum[1],extremum[2],extremum[3],extremum[4]
                                   if not self.__isEdge(i_octave,i,r,c,self.__CURV_THRESHOLD):
                                        keyPoints.append([xcor,ycor])

        return keyPoints


	#建立高斯金字塔
    def __buildGaussianPyramid(self, imgBase: np.ndarray, nOctaves: int, nIntervals: int, sigma: float) -> None:
        #k是每个interval的相对于前一interval的sigma的增量（通过乘k来实现增加）
        k = 2.0**(1/nIntervals)
        #这里建立的是一个octave上，每一层相对于上一层的sigma增量
        sigmas = [sigma, sigma * ((k*k-1)**0.5)]
        for i in range(2,nIntervals+3):
            sigmas.append(sigmas[-1]*k)
	#建立金字塔，对于同一octave上的多层，每一层都是由上一层经高斯模糊得来的
	#跨越octave,需要降采样时，直接使用上一octave倒数第三层的图像resize即可，这层图像和本octave第一层是同sigma的
        for i_octave in range(nOctaves):
            for i_interval in range(nIntervals+3):
                if i_octave == i_interval == 0:
                    self.gaussian_pyr.append([imgBase.copy()])
                elif i_interval == 0:
                    oriImg = self.gaussian_pyr[-1][-3]
                    self.gaussian_pyr.append([cv2.resize(oriImg,(oriImg.shape[0]//2, oriImg.shape[1]//2))])
                else:
                    self.gaussian_pyr[i_octave].append(cv2.GaussianBlur(self.gaussian_pyr[i_octave][-1],(0,0),sigmaX=sigmas[i_interval],sigmaY=sigmas[i_interval]))
	
	#建立DoG金字塔
    def __buildDoGPyramid(self, nOctaves: int, nIntervals: int):
        if not self.gaussian_pyr:
            raise Exception("Cannot calculate DoG pyramid before gaussian pyramid.")
        for i_octave in range(nOctaves):
            self.dog_pyr.append([])
            for i_interval in range(nIntervals+2):
                self.dog_pyr[-1].append(self.gaussian_pyr[i_octave][i_interval+1] - self.gaussian_pyr[i_octave][i_interval])

    #寻找局部的极值点
    def __isExtrema(self, i_octave: int, i_interval: int, row: int, col: int) -> bool:
        value = self.dog_pyr[i_octave][i_interval][row][col]
        if value > 0:
            for l in range(-1,2):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if self.dog_pyr[i_octave][i_interval+l][row+i][col+j] > value:
                            return False
        else:
            for l in range(-1,2):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if self.dog_pyr[i_octave][i_interval+l][row+i][col+j] < value:
                            return False
        return True

    #通过迭代插值寻找真正的极值点
    def __getInterpolatedExtremum(self,i_octave: int, i_interval: int, row: int, col: int, contr_Thr: float):
        nIntervals = len(self.dog_pyr[i_octave])
      
        width, height = self.dog_pyr[i_octave][i_interval].shape
        xc,xr,xi = .0,.0,.0
        #迭代通常最多5次
        for i in range(self.__MAX_INTERP_ITER_STEPS):
            xc,xr,xi = self.__interp_iter(i_octave,i_interval,row,col)
            if math.fabs(xc) < 0.5 and math.fabs(xr) < 0.5 and math.fabs(xi) < 0.5: #迭代已收敛
                break

            i_interval += round(xi)
            row += round(xr)
            col += round(xc)


            if i_interval<1 or i_interval >= nIntervals-1 or row < self.__SIFT_IMAGE_BORDER or col <self.__SIFT_IMAGE_BORDER \
            or row > height - self.__SIFT_IMAGE_BORDER or col > width - self.__SIFT_IMAGE_BORDER:
                return None
	
	#迭代未收敛，放弃极值点
        if i > self.__MAX_INTERP_ITER_STEPS:
            return None
	
	#计算插值后(即原极值点加上delta偏移量后)的对比度是否符合要求，不符则抛弃
        contra = self.__interp_contr(i_octave,i_interval,row,col,xc,xr,xi)
        if math.fabs(contra) < contr_Thr / (nIntervals-2):
            return None
	
	#返回关键点和它的尺度空间信息
        return (i_interval,row,col,(col + xc) * 2.0**i_octave, (row + xr) * 2.0**i_octave)

	#每次的迭代插值
    def __interp_iter(self,i_octave: int, i_interval: int, row: int, col: int) -> (float,float,float):
        dD = self.__getDerivD(i_octave,i_interval,row,col)
        H = self.__getHessianD(i_octave,i_interval,row,col)
        H_inv = np.linalg.inv(H)
        delta = np.matmul(-H_inv,dD)
        return (delta[0],delta[1],delta[2])
	#获取向量[dx,dy,dsigma]
    def __getDerivD(self,i_octave: int, i_interval: int, row: int, col: int) -> np.ndarray:
        #dx,dy,ds refer to the differential in x,y and sigma directions, which are calculated by difference
        dx = (self.dog_pyr[i_octave][i_interval][row][col+1] - self.dog_pyr[i_octave][i_interval][row][col-1]) / 2.0
        dy = (self.dog_pyr[i_octave][i_interval][row+1][col] - self.dog_pyr[i_octave][i_interval][row-1][col]) / 2.0
        ds = (self.dog_pyr[i_octave][i_interval][row+1][col] - self.dog_pyr[i_octave][i_interval][row-1][col]) / 2.0

        return np.array([dx,dy,ds])
        
	#获取D(x,y,sigma)的海森矩阵
    def __getHessianD(self,i_octave: int, i_interval: int, row: int, col: int) -> np.ndarray:

        v = self.dog_pyr[i_octave][i_interval][row][col]
        dxx = (self.dog_pyr[i_octave][i_interval][row][col+1] + self.dog_pyr[i_octave][i_interval][row][col-1] - 2*v )
        dyy = (self.dog_pyr[i_octave][i_interval][row+1][col] + self.dog_pyr[i_octave][i_interval][row-1][col] - 2*v )
        dss = (self.dog_pyr[i_octave][i_interval+1][row][col] + self.dog_pyr[i_octave][i_interval-1][row][col] - 2*v)


        dxy = (self.dog_pyr[i_octave][i_interval][row+1][col+1]  \
              + self.dog_pyr[i_octave][i_interval][row-1][col-1] \
              - self.dog_pyr[i_octave][i_interval][row+1][col-1] \
              - self.dog_pyr[i_octave][i_interval][row-1][col+1]) / 4.0

        dxs = (self.dog_pyr[i_octave][i_interval+1][row][col+1]  \
              + self.dog_pyr[i_octave][i_interval-1][row][col-1] \
              - self.dog_pyr[i_octave][i_interval+1][row][col-1] \
              - self.dog_pyr[i_octave][i_interval-1][row][col+1]) / 4.0

        dys = (self.dog_pyr[i_octave][i_interval+1][row+1][col]  \
              + self.dog_pyr[i_octave][i_interval-1][row-1][col] \
              - self.dog_pyr[i_octave][i_interval-1][row+1][col] \
              - self.dog_pyr[i_octave][i_interval+1][row-1][col]) / 4.0

        return np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
	#插值后的对比度
    def __interp_contr(self,i_octave: int, i_interval: int, row: int, col: int, xc: float, xr: float, xi: float) -> float:
        x = np.array([xc,xr,xi])
        dD = self.__getDerivD(i_octave,i_octave,row,col)
        t = np.matmul(dD,x)

        return self.dog_pyr[i_octave][i_interval][row][col] + t * 0.5
        
	#通过主曲率比值检验是否为边缘
    def __isEdge(self, i_octave: int, i_interval: int, row: int, col: int, curv_thr: int):
        d = self.dog_pyr[i_octave][i_interval][row][col]

        dxx = (self.dog_pyr[i_octave][i_interval][row][col+1] + self.dog_pyr[i_octave][i_interval][row][col-1] - 2*d)
        dyy = (self.dog_pyr[i_octave][i_interval][row+1][col] + self.dog_pyr[i_octave][i_interval][row-1][col] - 2*d)
        dxy = (self.dog_pyr[i_octave][i_interval][row+1][col+1] + self.dog_pyr[i_octave][i_interval][row-1][col-1] \
               -self.dog_pyr[i_octave][i_interval][row-1][col+1] - self.dog_pyr[i_octave][i_interval][row+1][col-1] )


        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy



        if det <= 0 or trace*trace/det >= (curv_thr + 1.0) * (curv_thr + 1.0) / curv_thr:
            return True
        return False


