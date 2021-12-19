# 5阶高斯卷积 



## 1. 代码实现关键点

这是一个5*5高斯卷积的python实现，卷积核由以下公式直接采样获取：
$$
f(x) = \dfrac{1}{2 \pi \sigma^2} e^{-\frac{(u^2+v^2)}{2\sigma^2}}
$$
​	此外，程序利用了高斯卷积的线性可分的特性，把5\*5的二维卷积拆分成了一个1\*5和一个5*1的卷积核，

它们依次在原图上做卷积，最后得到的图像和用5*5高斯核卷积后的一致，程序如下：

```
#两个一维高斯卷积，rowkernel即卷积核
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
                        sum += self.rowkernel[
```

​	这项改进使卷积效率获得了很大的提升。

​	

## 2. 测试结果



![ret](/home/scgao/myExecise/ImageEngineeringProjects/5阶高斯卷积/pictures/ori.jpg)![ret](/home/scgao/myExecise/ImageEngineeringProjects/5阶高斯卷积/pictures/ret.jpg)

**这两张图像中，左图为原图，右侧为高斯卷积处理后的图。**
