# DoG检测算子

​	DoG是SIFT特征检测的重要一部分，所以这个项目的许多细节都参考了SIFT的实现过程，主要是参考了项目[opensift](https://github.com/robwhess/opensift)，这个早期的开源代码最终被并入opencv库；理论部分，主要参考了[SIFT定位算法关键步骤的说明](https://www.cnblogs.com/ronny/p/4028776.html)。

​	这份代码注释不够规范，清谅解。

## 1. 代码实现关键点

### 1.1 高斯金字塔的构建

​		DoG金字塔基于高斯金字塔而建立，所以算法第一部是建立高斯金字塔$G(x,y,\sigma)$. 具体实现代码如下：

```
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
```

​	可以注意到，第二层相对于第一层的增量是$\sqrt{k^2-1}\sigma_0$而不是$k \sigma_0$，这是因为基于反走样的要求，我们要把原始图像看作已经被相机做过高斯模糊$\sigma_n$处理的图像，而且设定成把图像放大两倍之后的，$\sigma_n$通常摄制成0.5，所以增量为$\sqrt{k^2-2\sigma_n}\sigma_0 = \sqrt{k^2-1}\sigma_0$。这个部分可参考此[SIFT定位算法关键步骤的说明](https://www.cnblogs.com/ronny/p/4028776.html)。



### 1.2 DoG金字塔的构建

​	DoG金字塔根据高斯金字塔每个interval上的差值得来，代码如下：

```
	#建立DoG金字塔
    def __buildDoGPyramid(self, nOctaves: int, nIntervals: int):
        if not self.gaussian_pyr:
            raise Exception("Cannot calculate DoG pyramid before gaussian pyramid.")
        for i_octave in range(nOctaves):
            self.dog_pyr.append([])
            for i_interval in range(nIntervals+2):
                self.dog_pyr[-1].append(self.gaussian_pyr[i_octave][i_interval+1] - self.gaussian_pyr[i_octave][i_interval])
```



### 1.3 局部极值搜索

​	SIFT的局部极值定义为，在$D(x,y,sigma)$上一个3\*3\*3区域上，元素(x,y,sigma)取得的极大值或极小数值，代码如下：

```
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
```



### 1.4 插值获取精确极值点

#### 1.4.1  计算方法

​		在尺度空间获得的极值点映射回原始图像，不一定是真实的极值点，真实极值点可能存在于两个离散的极值点之间。为了获得精确的数据，有必要对离散点插值获取真正的极值点。

​	这部分首先将$D(x,y,\sigma)$做泰勒二阶展开，然后使它的一阶导数等于0，以此求得$\Delta x,\Delta y,\Delta  \sigma$三个变化量。具体如下，对于一般函数的二阶泰勒展开，有：
$$
f(x) \approx f(0) + f'(0)x + \frac{f''(0)}{2}x^2
$$
​	在离散的图像上，$f'(x)$和$f''(x)$可以由差分近似表示为：
$$
f'(x) & = &\frac{f(x+1)-f(x-1)}{2} \\
f''(x) & = &f(x+1) + f(x-1) - 2f(x)
$$
​	而对于多元的函数有：
$$
\therefore \begin{aligned}
\frac{\partial^{2} f\left(x, y\right)}{\partial x \partial y} \approx 
& \frac{1}{4 h^{2}}\left[f\left(x+h, y+h\right)+f\left(x-h, y-h\right)\right.
\\
&\left.-f\left(x+h, y-h\right)-f\left(x-h, y+h\right)\right]
\end{aligned}
$$
​	对于多变量的$D(x,y,\sigma)$，它的二阶泰勒展开式是:
$$
D(\Delta x,\Delta y,\Delta \sigma) = D(x,y,\sigma)+\begin{bmatrix}  \frac{\partial D}{x} & \frac{\partial D}{y} & \frac{\partial D}{\sigma} 

   \end{bmatrix}\begin{bmatrix} 

   \Delta x\\ 

   \Delta y\\ 

   \Delta \sigma 

   \end{bmatrix}+\frac{1}{2}\begin{bmatrix} 

   \Delta x &\Delta y  & \Delta \sigma 

   \end{bmatrix}\begin{bmatrix} 

   \frac{\partial ^2D}{\partial x^2} & \frac{\partial ^2D}{\partial x\partial y} &\frac{\partial ^2D}{\partial x\partial \sigma} \\ 

   \frac{\partial ^2D}{\partial y\partial x}& \frac{\partial ^2D}{\partial y^2} & \frac{\partial ^2D}{\partial y\partial \sigma}\\ 

   \frac{\partial ^2D}{\partial \sigma\partial x}&\frac{\partial ^2D}{\partial \sigma\partial y}  & \frac{\partial ^2D}{\partial \sigma^2} 

   \end{bmatrix}\begin{bmatrix} 

   \Delta x\\ 

   \Delta y\\ 

   \Delta \sigma 

   \end{bmatrix}
$$
写为向量形式：
$$
D(x) = D+\frac{\partial D^T}{\partial x}\Delta x+\frac{1}{2}\Delta x^T\frac{\partial ^2D^T}{\partial x^2}\Delta x
$$
使它的一阶导等于0，那么$\Delta x = -\frac{\partial^2D^{-1}}{\partial x^2}\frac{\partial D(x)}{\partial x}$。



#### 1.4.2 代码实现

​	参考opensift的代码，这样的近似插值是迭代计算的，最大迭代次数通常为5，若此时还不收敛，就抛弃这个极值点。

```
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
```



其中近似一阶导函数__getDerivD实现如下：

```
	#获取向量[dx,dy,dsigma]
    def __getDerivD(self,i_octave: int, i_interval: int, row: int, col: int) -> np.ndarray:
        #dx,dy,ds refer to the differential in x,y and sigma directions, which are calculated by difference
        dx = (self.dog_pyr[i_octave][i_interval][row][col+1] - self.dog_pyr[i_octave][i_interval][row][col-1]) / 2.0
        dy = (self.dog_pyr[i_octave][i_interval][row+1][col] - self.dog_pyr[i_octave][i_interval][row-1][col]) / 2.0
        ds = (self.dog_pyr[i_octave][i_interval][row+1][col] - self.dog_pyr[i_octave][i_interval][row-1][col]) / 2.0

        return np.array([dx,dy,ds])
```



而海森矩阵函数__getHessianD如下：

```
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
```



### 1.5 滤去边缘上的关键点

​	在图像中一些物体边缘上检测得到的关键点是不够稳定的，很难确保在其他条件下再次检测出该电，因此要去除边缘上的取点，这一步通过删去在某个方向的曲率明显很大的点来实现。

​	已知有海森矩阵：$H(x,y) = \begin{bmatrix}D_{xx}(x,y) & D_{xy}(x,y)\\ D_{xy}(x,y) &D_{yy}(x,y) \end{bmatrix}$

​	如果求得它的特征值$\lambda_\alpha,\lambda_\beta $，那么可以通过两个特征值的比率判断是否接近一个边缘。不过为了节省算力，可以通过海森矩阵的行列式$Det(H)$和迹$Tr(H)$来获得近似比值，假设两个$\lambda$的比值是$\gamma$,，通常$\gamma$取10：	
$$
\frac{Tr(H)^2}{Det(H)} = \frac{(\alpha+\beta)^2}{\alpha\beta} = \frac{(\gamma+1)^2}{\gamma}
$$
只需使上式小于预定$\gamma$的$\frac{(\gamma+1)^2}{\gamma}$即可。



## 2. 实现效果

​	对于文件夹中的buffterfly.png，不同对比度阈值获取如下结果：

![ret_1](/home/scgao/myExecise/ImageEngineeringProjects/DoG算子/pictures/ret_1.jpg)

![ret_2](/home/scgao/myExecise/ImageEngineeringProjects/DoG算子/pictures/ret_2.jpg)



​	上面两图分别是对比度阈值为0.1和0.04的版本，特征点显得多且杂乱，程序没有采用滤去响应值低于最大值0.1倍的点的策略。
