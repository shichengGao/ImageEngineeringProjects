import cv2
import numpy as np
import random
#compute Homography with four-point method
def getHomographyDLT(kp1,kp2):
    kp1.reshape(-1,2)
    kp2.reshape(-1,2)
    assert len(kp1) == len(kp2) == 4
    A = np.array([
        [kp1[0][0], kp1[0][1], 1, 0, 0, 0, -kp1[0][0] * kp2[0][0], -kp1[0][1] * kp2[0][0]],
        [0, 0, 0, kp1[0][0], kp1[0][1], 1, -kp1[0][0] * kp2[0][1], -kp1[0][1] * kp2[0][1]],
        [kp1[1][0], kp1[1][1], 1, 0, 0, 0, -kp1[1][0] * kp1[1][0], -kp1[1][1] * kp1[1][0]],
        [0, 0, 0, kp1[1][0], kp1[1][1], 1, -kp1[1][0] * kp1[1][1], -kp1[1][1] * kp1[1][1]],
        [kp1[2][0], kp1[2][1], 1, 0, 0, 0, -kp1[2][0] * kp1[2][0], -kp1[2][1] * kp1[2][0]],
        [0, 0, 0, kp1[2][0], kp1[2][1], 1, -kp1[2][0] * kp1[2][1], -kp1[2][1] * kp1[2][1]],
        [kp1[3][0], kp1[3][1], 1, 0, 0, 0, -kp1[3][0] * kp1[3][0], -kp1[3][1] * kp1[3][0]],
        [0, 0, 0, kp1[3][0], kp1[3][1], 1, -kp1[3][0] * kp1[3][1], -kp1[3][1] * kp1[3][1]]
    ])
    b = kp2.reshape(-1,1)
    h = np.linalg.solve(A,b)
    h = np.append(h,1.0)
    return h.reshape(-1,3)

#computer homography with RANSAC(random sample consensus)
def findHomography_RANSAC(kp1:np.ndarray, kp2:np.ndarray, ransac_rpr_thr:np.float32 , maxIter=30):
    assert len(kp1) == len(kp2)
    total_points_num = len(kp1)
    iter = 0
    max_inliers_num = 0
    result_H = None
    while iter < maxIter and max_inliers_num<0.8*total_points_num:
        #select four pairs of points
        indices =  np.random.choice(a=total_points_num,size=4,replace=False)
        src_points = np.array([kp1[i] for i in indices])
        dst_points = np.array([kp2[i] for i in indices])

        #compute homography by these points
        H = getHomographyDLT(src_points,dst_points)
        inliers_num = 0
        for i in range(total_points_num):
            src_vec = np.array([kp1[i][0],kp1[i][1],1.0])
            dst_vec = np.matmul(H,src_vec)
            if ((kp2[i][0]-dst_vec[0])**2 + (kp2[i][1]-dst_vec[1])**2)**0.5 < ransac_rpr_thr:
                inliers_num += 1
        if inliers_num > max_inliers_num:
            max_inliers_num, result_H = inliers_num, H

    return result_H