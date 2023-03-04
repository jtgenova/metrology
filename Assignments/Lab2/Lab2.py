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
    delta_X = 10254.288862572017
    delta_Y = -10378.775192131347
    A = 84.0376645345848
    B = -0.0021168248123863975
    C = 0.00946550579867278
    D = 84.02468309158996
    A_mat = np.array([[A, B], [C, D]])

    xc = [-105.997, 106.004, -106, 106.012, -112, 112.006, 0.005, 0.002]
    yc = [-105.995, 106.008, 106.009, -105.995, 0.007, 0.007, 112.007, -111.998]
    img1_fid_x, img1, fid_y = get_fiducial(xc, yc, A_mat, delta_X, delta_Y)
