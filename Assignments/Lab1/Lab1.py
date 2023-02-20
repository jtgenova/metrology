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
    """
    Similarity Transform
    Find translation (delta x, delta y)
    Find scale (lambda)
    Find rotation (theta)
    """
    unknowns = 32
    l_mat = np.zeros(shape=(unknowns,1))
    idx = 0
    for i in range(0, unknowns, 2):
        l_mat[i] = xf[idx]
        l_mat[i+1] = yf[idx]
        idx += 1
    
    A_mat = np.zeros(shape=(unknowns,4))
    idx = 0
    for i in range(0, unknowns, 2):
        A_mat[i] = [xc[idx], -yc[idx], 1, 0]
        A_mat[i+1] = [yc[idx], xc[idx], 0, 1]
        idx +=1 
    # print(A_mat)

    x_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A_mat),A_mat)), np.transpose(A_mat)), l_mat)
    A, B, Dx, Dy = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3])
    scale = math.sqrt(A**2 + B**2)
    theta = math.atan(B/A)
    
    # print(x_hat)
    print(f'A: {A}')
    print(f'B: {B}')
    print(f'delta X: {Dx}')
    print(f'delta Y: {Dy}')
    print(f'scale: {scale}')
    print(f'theta: {theta}')
    
    v = np.dot(A_mat, x_hat) - l_mat
    unknowns = 32
    v_mat = np.zeros(shape=(16,2))
    idx = 0
    for i in range(0, 16, 2):
        l_mat[i] = xf[idx]
        l_mat[i+1] = yf[idx]
        idx += 1
    print(v)
    return 
    # return Dx, Dy, scale, theta

def affine_transform(xc, yc, xf, yf):
    """
    Affine Transform
    Find translation (delta x, delta y)
    Find rotation (theta)
    Find scale (Sx, Sy)
    Find non-orthogonality of comparator axes (delta)
    """
    unknowns = 6
    l_mat = np.zeros(shape=(unknowns,1))
    idx = 0
    for i in range(0, unknowns, 2):
        l_mat[i] = xf[idx]
        l_mat[i+1] = yf[idx]
        idx += 1

    A_mat = np.zeros(shape=(unknowns,6))
    idx = 0
    for i in range(0, unknowns, 2):
        A_mat[i] = [xc[idx], -yc[idx], 1, 0, 0 ,0]
        A_mat[i+1] = [0, 0, 0, yc[idx], xc[idx], 1]
        idx +=1 

    x_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A_mat),A_mat)), np.transpose(A_mat)), l_mat)
    A, B, Dx, C, D, Dy = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3]), float(x_hat[4]), float(x_hat[5])
    theta = math.atan(C/A)
    Sx = math.sqrt(A**2 + C**2)
    Sy = math.sqrt(B**2 + D**2)
    delta =math.atan((A*B + C*D)/(A*D - B*C))

    # print(f'delta X: {Dx}')
    # print(f'delta Y: {Dy}')
    # print(f'theta: {theta}')
    # print(f'Scale X: {Sx}')
    # print(f'Scale Y: {Sy}')
    # print(f'delta: {delta}')

    v = np.dot(A_mat, x_hat) - l_mat
    print(v)

    return Dx, Dy, theta, Sx, Sy, delta

def projective_trans(xc, yc, xf, yf):
    """
    Projective Transform
    Find translation (delta x, delta y)
    Find differential scale (Sx, Sy)
    Find non-orthogonality (delta)
    Find rotation (theta)
    Find out of plane inclination (2 parameters)
    """
    return

if __name__=="__main__":
    # test vals
    xc = [-113.767, -43.717, 36.361, 106.408, 107.189, 37.137, -42.919, -102.968, -112.052, -42.005, 38.051, 108.089, 108.884, 38.846, -41.208, -111.249]
    yc = [-107.400, -108.204, -109.132, -109.923, -39.874, -39.070, -38.158, -37.446, 42.714, 41.903, 40.985, 40.189, 110.221, 111.029, 111.961, 112.759]
    xf = [-110, -40, 40, 110, 110, 40, -40, -100, -110, -40, 40, 110, 110, 40, -40, -110]
    yf = [-110, -110, -110, -110, -40, -40, -40, -40, 40, 40, 40, 40, 110, 110, 110, 110]

    # assignment vals

    sim_Dx, sim_Dy, sim_scale, sim_theta = similarity_transform(xc, yc, xf, yf)
    # matplotlib.pyplot.quiver(xc, yc, xf, yf)
    # aff_Dx, aff_Dy, aff_theta, aff_Sx, aff_Sy, aff_delta = affine_transform(xc, yc, xf, yf)


