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
    print(f'Total Correction: \n {correction}\n')

##############################################################################################################################################################
    # Image 2
    delta_X = -122.19211044565897
    delta_Y = 123.51804729053579
    A = 0.011900088285313318
    B = -8.456447779614914e-06
    C = 7.403491422692827e-06
    D = 0.011901033060072988
    A_mat = np.array([[A, B], [C, D]])

    xc2 = [1411, 9416, 2275, 11129, 4160, 10137]
    yc2 = [-2081, -1167, -10787, -10048, -17085, -17690]

    idx = 0
    correction = np.zeros(shape=(len(xc2),2))
    fid_coords = np.zeros(shape=(len(xc2),2))
    for i in range(len(xc2)):
        # get fiducial coordinates
        xf, yf = pre_lab_3.get_fiducial(xc2[i], yc2[i], A_mat, delta_X, delta_Y)
        fid_coords[idx][0] = xf
        fid_coords[idx][1] = yf

        x_total, y_total = pre_lab_3.get_total(xf, yf, principal_point_offset, radial_lens_distortion, decentering_distortion, focal_length, k_atmos)
        correction[idx][0] = x_total
        correction[idx][1] = y_total
        idx += 1
    print(f'Total Correction: \n {correction}\n')