"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 3
Description: Relative Orientation
Deadline: March 22, 2023 10:00 AM
"""

import numpy as np
from numpy.linalg import inv, det
import math
import matplotlib.pyplot as plt
import preLab3 as pre_lab_3

def correct_images(xc1, yc1, xc2, yc2):
    idx = 0
    correction_left = np.zeros(shape=(len(xc1),2))
    correction_right = np.zeros(shape=(len(xc2),2))
    for i in range(len(xc1)):
        # correct images 1
        xl, yl = pre_lab_3.get_left(xc1[i], yc1[i])
        correction_left[idx][0] = round(xl, 3)
        correction_left[idx][1] = round(yl, 3)

        # correct images 2
        xr, yr = pre_lab_3.get_left(xc2[i], yc2[i])
        correction_right[idx][0] = round(xr, 3)
        correction_right[idx][1] = round(yr, 3)
        idx += 1

    # print(f'Total Correction Left: \n {correction_left}\n')
    # print(f'Total Correction Right: \n {correction_right}\n')
    return correction_left, correction_right


def find_A_elems(xl, yl, c, xr, yr, zr, bx, by, bz, omega, phi, kappa):

    dby = np.array([[0, 1, 0], [xl, yl, -c], [xr, yr, zr]])
    # print(f'dby:\n {dby}')
    dby = det(dby)
    # print(f'dby = {dby}')

    dbz = np.array([[0, 0, 1], [xl, yl, -c], [xr, yr, zr]])
    # print(f'dbz:\n {dbz}')
    dbz = det(dbz)
    # print(f'dbz = {dbz}')

    dw = np.array([[bx, by, bz], [xl, yl, -c], [0, -zr, yr]])
    # print(f'dw:\n {dw}')
    dw = det(dw)
    # print(f'dw = {dw}')

    A = -yr*math.sin(omega) + zr*math.cos(omega)
    B = xr*math.sin(omega)
    C = -xr*math.cos(omega)
    dphi = np.array([[bx, by, bz], [xl, yl, -c], [A, B, C]])
    # print(f'dphi:\n {dphi}')
    dphi = det(dphi)
    # print(f'dphi = {dphi}')

    D = -yr*math.cos(omega)*math.cos(phi) - zr*math.sin(omega)*math.cos(phi)
    E = xr*math.cos(omega)*math.cos(phi) - zr*math.sin(phi)
    F = xr*math.sin(omega)*math.cos(phi) + yr*math.sin(phi)
    dkappa = np.array([[bx, by, bz], [xl, yl, -c], [D, E, F]])
    # print(f'dkappa:\n {dkappa}')
    dkappa = det(dkappa)
    # print(f'dkappa = {dkappa}')

    return dby, dbz, dw, dphi, dkappa

def find_misclosure(xl, yl, c, xr, yr, bx, by, bz):
    zr = -c

    w = np.array([[bx, by, bz], [xl, yl, -c], [xr, yr, zr]])
    w = det(w)

    return w

def find_delta(xl, yl, c, xr, yr, bx, by, bz, omega, phi, kappa):
    A_matrix = np.zeros(shape=(len(xl),5))
    w = np.zeros(shape=(len(xl), 1))
    idx = 0
    for i in range(len(xl)):
        dby, dbz, dw, dphi, dkappa = find_A_elems(xl[i], yl[i], c, xr[i], yr[i], -c, bx, by, bz, omega, phi, kappa)
        A_matrix[idx][0] = dby
        A_matrix[idx][1] = dbz
        A_matrix[idx][2] = dw
        A_matrix[idx][3] = dphi
        A_matrix[idx][4] = dkappa
        w[idx] = find_misclosure(xl[i], yl[i], c, xr[i], yr[i], bx, by=0, bz=0)

        idx += 1
    A_matrix_trans = np.transpose(A_matrix)
    by, bz, omega, phi, kappa = -np.dot(np.dot(inv(np.dot(A_matrix_trans, A_matrix)), A_matrix_trans), w)
    
    return by[0], bz[0], omega[0], phi[0], kappa[0]

def space_intersection(xl, yl, c, xr, yr, bx, by, bz):
    zr = -c
    scale = (bx*zr - bz*xr) / (xl*zr - c*xr)
    mu = (-bx*c - bz*xl) / (xl*zr + c*xr)
    print(f'scale for left: {scale}')
    print(f'scale for right: {mu}')

    model_Xl = scale*xl
    model_Yl = scale*yl
    model_Zl = -scale*c

    model_Xr = mu*xr + bx
    model_Yr = mu*yr + by
    model_Zr = mu*zr + bz

    model_L = np.transpose(np.array([model_Xl, (model_Yl + model_Yr)/2, model_Zl]))
    model_R = np.transpose(np.array([model_Xr, (model_Yl + model_Yr)/2, model_Zr]))

    # print(f'Model L:\n {model_L}')
    # print(f'Model R:\n {model_R}')
    pY = model_Yr - model_Yl
    # print(f'y-parallax values: {pY}')

    return model_L, model_R, pY, scale, mu

def plot_scale(scale_left, scale_right):
    id = np.array([100, 101, 102, 103, 104, 105])
    plt.subplot(1,2,1)
    plt.bar(id, scale_left, color='darkblue')
    plt.xlabel("Image ID", fontdict={'family':'serif','color':'black','size':10})
    plt.ylabel('Scale Factor (λ)', fontdict={'family':'serif','color':'black','size':10})
    plt.title("Left Image Scale Factor", fontdict ={'family':'serif','color':'black','size':12})

    plt.subplot(1,2,2)
    plt.bar(id, scale_right, color='darkred')
    plt.xlabel("Image ID", fontdict={'family':'serif','color':'black','size':10})
    plt.ylabel('Scale Factor(μ)', fontdict={'family':'serif','color':'black','size':10})
    plt.title("Right Image Scale Factor", fontdict ={'family':'serif','color':'black','size':12})

    plt.show()


##############################################################################################################################################################
if __name__=="__main__":

    # Image 1
    xc1 = [9460, 17400, 10059, 19158, 11844, 17842]
    yc1 = [-2292, -1661, -10883, -10412, -17253, -18028]

    # Image 2
    xc2 = [1411, 9416, 2275, 11129, 4160, 10137]
    yc2 = [-2081, -1167, -10787, -10048, -17085, -17690]

    # bx
    bx = 92.000

    left_images, right_images = correct_images(xc1, yc1, xc2, yc2)
    xl = left_images[:,0]
    yl = left_images[:,1]
    xr = right_images[:,0]
    yr = right_images[:,1]

    # idx = 0

    # A_matrix = np.zeros(shape=(len(xf),5))
    # for i in range(len(xf)):
    #     dby, dbz, dw, dphi, dkappa = find_A_elems(0, 0, c, xl[i], yl[i], c, bx)
    #     A_matrix[idx][0] = dby
    #     A_matrix[idx][1] = dbz
    #     A_matrix[idx][2] = dw
    #     A_matrix[idx][3] = dphi
    #     A_matrix[idx][4] = dkappa
    # print(f'A matrix = {A_matrix}')

###################################################
    # Example
    c = 152.15
    bx = 92.000
    xl = np.array([106.399, 18.989, 70.964, -0.931, 9.278, 98.681])
    yl = np.array([90.426, 93.365, 4.907, -7.284, -92.926, -62.769])
    xr = np.array([24.848, -59.653, -15.581, -85.407, -78.81, 8.492])
    yr = np.array([81.824, 88.138, -0.387, -8.351, -92.62, -68.873])

    # dby, dbz, dw, dphi, dkappa = find_A_elems(xl[0], yl[0], c, xr[0], yr[0], -c, bx, by=0, bz=0, omega=0, phi=0, kappa=0)
    iter = 1
    by, bz, omega, phi, kappa = find_delta(xl, yl, c, xr, yr, bx, by=0, bz=0, omega=0, phi=0, kappa=0)
    for i in range(3):
        iter += 1
        by, bz, omega, phi, kappa = find_delta(xl, yl, c, xr, yr, bx, by, bz, omega, phi, kappa)
    # print(f'Number of iterations = {iter}')
    # print(f'delta:\n {np.array([by, bz, omega, phi, kappa])}')

    model_L, model_R, pY, scale_left, scale_right = space_intersection(xl, yl, c, xr, yr, bx, by, bz)

    plot_scale(scale_left, scale_right)

        
    