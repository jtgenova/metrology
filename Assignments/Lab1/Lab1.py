"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 1
Description: Similarity, Affine and Projective Transformations
Deadline: February 22, 2023 10:00 AM
"""
import time
import numpy as np
import math

def similarity_transform(xc, yc, xf, yf):
    l =np.array([[xf[0]],
                [yf[0]],
                [xf[1]],
                [yf[1]]])
    A_mat = np.array([[xc[0], -yc[0], 1, 0],
         [yc[0], xc[0], 0, 1],
         [xc[1], -yc[1], 1, 0],
         [yc[1], xc[1], 0, 1]])
    identity = np.identity(4)

    x_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A_mat),A_mat)), np.transpose(A_mat)), l)
    A, B, Dx, Dy = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])
    scale = math.sqrt(A**2 + B**2)
    theta = math.atan(B/A)
    return A, B, Dx, Dy, scale, theta


if __name__=="__main__":
    xc = [-113.767, -43.717]
    yc = [-107.4, -108.204]
    xf = [-110, -40]
    yf = [-110, -110]

    sim_A, sim_B, sim_Dx, sim_Dy, sim_scale, sim_theta = similarity_transform(xc, yc, xf, yf)
