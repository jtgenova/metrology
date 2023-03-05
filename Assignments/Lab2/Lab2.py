"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 1
Description: Similarity, Affine and Projective Transformations
Deadline: February 22, 2023 10:00 AM
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def get_fiducial(xc, yc, A_mat, x_delta, y_delta):
    idx = 0
    xf = np.zeros(len(xc))
    yf = np.zeros(len(yc))
    delta_xy = np.array([[x_delta], [y_delta]])
    for i in range(len(xc)):
        temp_c = np.array([[xc[i]], [yc[i]]])
        xf[idx], yf[idx] = np.dot(A_mat, temp_c) + delta_xy
        idx += 1
    return  xf, yf

if __name__=="__main__":
    
    # Given from calibration certificate
    focal_length = 153.358 # mm
    principal_point_offset = [-0.006, 0.006] # [xp, yp] mm
    radial_lens_distortion = [-0.8878e-4, -0.1528e-7, 0.5256e-12, 0.0000, 0.0000] # [K0, K1, K2, K3, K4]
    decentering_distortion = [0.1346e-06, 0.1224e-07, 0.0000, 0.0000] # [P1, P2, P3, P4]
    # Given from handout
    elevation = 1860 # m
    average_ground_elevation = 1100 # m
    scale_number = 5000
    image_size = 9 # in square
    # Given from affine transformation:
    delta_X = -122.01704301790505
    delta_Y = 123.53429666924897
    A = 0.011899426266928173
    B = 2.9976774439538446e-07
    C = -1.3405013290104452e-06
    D = 0.011901264695956251
    A_mat = np.array([[A, B], [C, D]])

    # xc = [-105.997, 106.004, -106, 106.012, -112, 112.006, 0.005, 0.002]
    # yc = [-105.995, 106.008, 106.009, -105.995, 0.007, 0.007, 112.007, -111.998]
    

    xc1 = [9460, 17400, 10059, 19158, 11844, 17842, 11781, 14009, 9617, 14686]
    yc1 = [-2292, -1661, -10883, -10412, -17253, -18028, -1174, -9749, -14502, -18204]

    xc2 = [1411, 9416, 2275, 11129, 4160, 10137, 3726, 6161, 1954, 6984]
    yc2 = [-2081, -1167, -10787, -10048, -17085, -17690, -854, -9528, -14416, -17948]


    img1_fid_x, img1_fid_y = get_fiducial(xc1, yc1, A_mat, delta_X, delta_Y)
    print(img1_fid_x.T)
