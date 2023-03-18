"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 3
Description: Relative Orientation
Deadline: March 22, 2023 10:00 AM
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import preLab3 as pre_lab_3

def find_A_elems(xl, yl, c, xr, yr, zr, bx):
    dby = np.array([[0, 1, 0], [xl, yl, -c], [xr, yr, zr]])
    dby = np.linalg.det(dby)
    print(f'dby = {dby}')

    dbz = np.array([[0, 0, 1], [xl, xr, -c], [xr, yr, zr]])
    dbz = np.linalg.det(dbz)
    print(f'dbz = {dbz}')

    dw = np.array([[bx, by, bz], [xl, yl, -c], [0, -zr, yr]])
    dw = np.linalg.det(dw)
    print(f'dw = {dw}')
    
    by = 0
    bz = 0
    w = 0

    A = -yr*math.sin(w) + zr*math.cos(w)
    B = xr*math.sin(w)
    C = -xr*math.cos(w)
    dphi = np.array([[bx, by, bz], [xl, yl, -c], [A, B, C]])
    dphi = np.linalg.det(dphi)
    print(f'dphi = {dphi}')

    phi = 0

    D = -yr*math.cos(w)*math.cos(phi) - zr*math.sin(w)*math.cos(phi)
    E = xr*math.cos(w)*math.cos(phi) - zr*math.sin(phi)
    F = xr*math.sin(w)*math.cos(phi) + yr*math.sin(phi)
    dkappa = np.array([[bx, by, bz], [xl, yl, -c], [D, E, F]])
    dkappa = np.linalg.det(dkappa)
    print(f'dkappa = {dkappa}')

    return dby, dbz, dw, dphi, dkappa

def space_intersection(B, L, R):
    bx = B[0]
    by = B[1]
    bz = B[2]
    xl = L[0]
    yl = L[1]
    c = L[2]
    xr = R[0]
    yr = R[1]
    zr = R[2]

    scale = (bx*zr - bz*xr) / (xl*zr - c*xr)
    mu = (-bx*c - bz*xl) / (xl*zr + c*xr)

    model_Xl = scale*xl
    model_Yl = scale*yl
    model_Zl = -scale*c

    model_Xr = mu*xr + bx
    model_Yr = mu*yr + by
    model_Zr = mu*zr + bz

    model_L = np.transpose(np.array([model_Xl, (model_Yl + model_Yr)/2, model_Zl]))
    model_R = np.transpose(np.array([model_Xr, (model_Yl + model_Yr)/2, model_Zr]))

    pY = model_Yr - model_Yl

##############################################################################################################################################################
if __name__=="__main__":

    # Given from calibration certificate
    focal_length = 153.358 # mm
    principal_point_offset = [-0.006, 0.006] # [xp, yp] mm
    radial_lens_distortion = [0.8878e-4, -0.1528e-7, 0.5256e-12] # [K0, K1, K2]
    decentering_distortion = [0.1346e-06, 0.1224e-07] # [P1, P2]
    c = focal_length # speed of light
    # Given from handout
    H = 1860/1000 # [km] elevation
    h = 1100/1000 # [km] ground elevation
    scale_number = 5000
    image_size = 9 # in square
    k_atmos = ((2410*H)/(H**2 -6*H + 250) - (2410*h)/(h**2 - 6*h + 250)*(h/H))*1e-6

    # bx
    bx = 92.000
   
    # Image 1
    delta_X = -122.01704301790505
    delta_Y = 123.53429666924897
    A = 0.011899426266928175
    B = 0.000000299767744395384
    C = -0.00000134050132901044
    D = 0.011901264695956251
    A_mat = np.array([[A, B], [C, D]])

    xc1 = [9460, 17400, 10059, 19158, 11844, 17842]
    yc1 = [-2292, -1661, -10883, -10412, -17253, -18028]

    idx = 0
    correction = np.zeros(shape=(len(xc1),2))
    fid_coords = np.zeros(shape=(len(xc1),2))
    for i in range(len(xc1)):
        # get fiducial coordinates
        xf, yf = pre_lab_3.get_fiducial(xc1[i], yc1[i], A_mat, delta_X, delta_Y)
        fid_coords[idx][0] = xf
        fid_coords[idx][1] = yf

        x_total, y_total = pre_lab_3.get_total(xf, yf, principal_point_offset, radial_lens_distortion, decentering_distortion, focal_length, k_atmos)
        correction[idx][0] = x_total
        correction[idx][1] = y_total
        idx += 1
    # print(f'Total Correction: \n {correction}\n')
    xf = correction[:,0]
    yf = correction[:,1]
    idx = 0

    A_matrix = np.zeros(shape=(len(xf),5))
    for i in range(len(xf)):
        dby, dbz, dw, dphi, dkappa = find_A_elems(0, 0, c, xf[i], yf[i], c, bx)
        A_matrix[idx][0] = dby
        A_matrix[idx][1] = dbz
        A_matrix[idx][2] = dw
        A_matrix[idx][3] = dphi
        A_matrix[idx][4] = dkappa
    print(f'A matrix = {A_matrix}')

    



##############################################################################################################################################################
    # # Image 2
    # delta_X = -122.19211044565897
    # delta_Y = 123.51804729053579
    # A = 0.011900088285313318
    # B = -8.456447779614914e-06
    # C = 7.403491422692827e-06
    # D = 0.011901033060072988
    # A_mat = np.array([[A, B], [C, D]])

    # xc2 = [1411, 9416, 2275, 11129, 4160, 10137]
    # yc2 = [-2081, -1167, -10787, -10048, -17085, -17690]

    # idx = 0
    # correction = np.zeros(shape=(len(xc2),2))
    # fid_coords = np.zeros(shape=(len(xc2),2))
    # for i in range(len(xc2)):
    #     # get fiducial coordinates
    #     xf, yf = pre_lab_3.get_fiducial(xc2[i], yc2[i], A_mat, delta_X, delta_Y)
    #     fid_coords[idx][0] = xf
    #     fid_coords[idx][1] = yf

    #     x_total, y_total = pre_lab_3.get_total(xf, yf, principal_point_offset, radial_lens_distortion, decentering_distortion, focal_length, k_atmos)
    #     correction[idx][0] = x_total
    #     correction[idx][1] = y_total
    #     idx += 1
    # print(f'Total Correction: \n {correction}\n')