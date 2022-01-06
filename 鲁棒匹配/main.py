import numpy as np
import cv2
import RANSAC

def main():
	#选取一张图像，把他旋转顺时针90度
    ori_img = cv2.imread('./owl.jpg')
    I1 = cv2.resize(ori_img, (ori_img.shape[1]//4,ori_img.shape[0]//4))
    I2 = cv2.rotate(I1, cv2.ROTATE_90_CLOCKWISE)
	#SIFT特征提取和匹配
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(I1, None)
    kp2, desc2 = sift.detectAndCompute(I2, None)

    kp_image1 = cv2.drawKeypoints(I1, kp1, None)
    kp_image2 = cv2.drawKeypoints(I2, kp2, None)

    mather = cv2.BFMatcher()
    raw_matches = mather.knnMatch(desc1,desc2,2)
    good_matches = []

    for m,n in raw_matches:
        if m.distance < 0.7*n.distance:
            good_matches.append([m])
    matches = cv2.drawMatchesKnn(I1,kp1,I2,kp2,good_matches, None, flags=2)
	#使用RANSAC模块进行参数估计，获取单应矩阵H
    dst_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    src_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    H = RANSAC.findHomography_RANSAC(src_pts,dst_pts,5.0)
    print('the estimated homography is :',H)
    	#根据H，修正图像，然后和原图进行对比
    I2 = cv2.warpPerspective(I2,H,(I2.shape[0],I2.shape[1]))
    cv2.imshow("matches",matches)
    cv2.imshow("original image",I1)
    cv2.imshow("the image corrected according to H after rotation",I2)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
