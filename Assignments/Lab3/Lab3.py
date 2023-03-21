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

def find_A_elems(xl, yl, c, xr, yr, zr, bx, by, bz, omega, phi, kappa):

    dby = np.array([[0, 1, 0], [xl, yl, -c], [xr, yr, zr]])
    dby = np.linalg.det(dby)
    # print(f'dby = {dby}')

    dbz = np.array([[0, 0, 1], [xl, yl, -c], [xr, yl, zr]])
    dbz = np.linalg.det(dbz)
    # print(f'dbz = {dbz}')

    dw = np.array([[bx, by, bz], [xl, yl, -c], [0, -zr, yr]])
    dw = np.linalg.det(dw)
    # print(f'dw = {dw}')

    A = -yr*math.sin(omega) + zr*math.cos(omega)
    B = xr*math.sin(omega)
    C = -xr*math.cos(omega)
    dphi = np.array([[bx, by, bz], [xl, yl, -c], [A, B, C]])
    dphi = np.linalg.det(dphi)
    # print(f'dphi = {dphi}')

    D = -yr*math.cos(omega)*math.cos(phi) - zr*math.sin(omega)*math.cos(phi)
    E = xr*math.cos(omega)*math.cos(phi) - zr*math.sin(phi)
    F = xr*math.sin(omega)*math.cos(phi) + yr*math.sin(phi)
    dkappa = np.array([[bx, by, bz], [xl, yl, -c], [D, E, F]])
    dkappa = np.linalg.det(dkappa)
    # print(f'dkappa = {dkappa}')

    return dby, dbz, dw, dphi, dkappa

def find_misclosure(xl, yl, c, xr, yr, bx, by, bz):
    zr = -c

    w = np.array([[bx, by, bz], [xl, yl, -c], [xr, yr, zr]])
    w = np.linalg.det(w)

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
    delta = -np.dot(np.dot(np.linalg.inv(np.dot(A_matrix_trans, A_matrix)), A_matrix_trans), w)
    # print(f'A matrix = {A_matrix}')
    # print(w)
    print(delta)
    return delta



def space_intersection(xl, yl, c, xr, yr, zr, bx, by, bz):

    scale = (bx*zr - bz*xr) / (xl*zr - c*xr)
    mu = (-bx*c - bz*xl) / (xl*zr + c*xr)
    # print(f'scale for left: {scale}')
    # print(f'scale for right: {mu}')

    model_Xl = scale*xl
    model_Yl = scale*yl
    model_Zl = -scale*c

    model_Xr = mu*xr + bx
    model_Yr = mu*yr + by
    model_Zr = mu*zr + bz

    model_L = np.transpose(np.array([model_Xl, (model_Yl + model_Yr)/2, model_Zl]))
    model_R = np.transpose(np.array([model_Xr, (model_Yl + model_Yr)/2, model_Zr]))

    # print(f'Model L: {model_L}')
    # print(f'Model R: {model_R}')
    pY = model_Yr - model_Yl
    # print(f'y-parallax values: {pY}')

    # return model_L, model_R, pY

# def model_space(xl, yl, c, xr, yr, zr, bx, by, bz):


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
    delta_X1 = -122.01704301790505
    delta_Y1 = 123.53429666924897
    A1 = 0.011899426266928175
    B1 = 0.000000299767744395384
    C1 = -0.00000134050132901044
    D1 = 0.011901264695956251
    A_mat1 = np.array([[A1, B1], [C1, D1]])

    xc1 = [9460, 17400, 10059, 19158, 11844, 17842]
    yc1 = [-2292, -1661, -10883, -10412, -17253, -18028]

    # Image 2
    delta_X2 = -122.19211044565897
    delta_Y2 = 123.51804729053579
    A2 = 0.011900088285313318
    B2 = -8.456447779614914e-06
    C2 = 7.403491422692827e-06
    D2 = 0.011901033060072988
    A_mat2 = np.array([[A2, B2], [C2, D2]])

    xc2 = [1411, 9416, 2275, 11129, 4160, 10137]
    yc2 = [-2081, -1167, -10787, -10048, -17085, -17690]

    idx = 0
    correction_left = np.zeros(shape=(len(xc1),2))
    correction_right = np.zeros(shape=(len(xc2),2))
    for i in range(len(xc1)):
        # get fiducial coordinates 1
        xf, yf = pre_lab_3.get_fiducial(xc1[i], yc1[i], A_mat1, delta_X1, delta_Y1)
        # correct images 1
        xl, yl = pre_lab_3.get_total(xf, yf, principal_point_offset, radial_lens_distortion, decentering_distortion, focal_length, k_atmos)
        correction_left[idx][0] = round(xl, 3)
        correction_left[idx][1] = round(yl, 3)

        # get fiducial coordinates 2
        xf, yf = pre_lab_3.get_fiducial(xc2[i], yc2[i], A_mat2, delta_X2, delta_Y2)
        # correct images 2
        xr, yr = pre_lab_3.get_total(xf, yf, principal_point_offset, radial_lens_distortion, decentering_distortion, focal_length, k_atmos)
        correction_right[idx][0] = round(xr, 3)
        correction_right[idx][1] = round(yr, 3)
        idx += 1
    # print(f'Total Correction Left: \n {correction_left}\n')
    # print(f'Total Correction Right: \n {correction_right}\n')
    idx = 0

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

    # estimated params
    by = 0
    bz = 0
    omega = 0
    phi = 0
    kappa = 0

    iter = 1
    print(f'Number of iterations = {iter}')
    delta = find_delta(xl, yl, c, xr, yr, bx, by, bz, omega, phi, kappa)
    by = delta[0][0]
    bz = delta[1][0]
    omega = delta[2][0]
    phi = delta[3][0]
    kappa = delta[4][0] 
    for i in range(3):
        iter += 1
        print(f'Number of iterations = {iter}')
        delta = find_delta(xl, yl, c, xr, yr, bx, by, bz, omega, phi, kappa)
   
    by = delta[0][0]
    bz = delta[1][0]
    omega = delta[2][0]
    phi = delta[3][0]
    kappa = delta[4][0]    
    space_intersection(xl, yl, c, xr, yr, -c, bx, by, bz)
        
    