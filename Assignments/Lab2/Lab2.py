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
    fid_coords = np.zeros(shape=(len(xc),2))
    delta_xy = np.array([[x_delta], [y_delta]])
    for i in range(len(xc)):
        temp_c = np.array([[xc[i]], [yc[i]]])
        xf[idx], yf[idx] = np.dot(A_mat, temp_c) + delta_xy
        fid_coords[idx][0] = xf[idx]
        fid_coords[idx][1] = yf[idx]
        idx += 1
    print(f'Fiduciary Coordinates: \n{fid_coords}\n')
    return  fid_coords

def get_pp(fid_coords, pp):
    idx = 0
    fid_coords_pp = np.zeros(shape=(len(fid_coords),2))
    r_vals = np.zeros(shape=(len(fid_coords),1))
    for i in range(len(fid_coords)):
        x_temp = fid_coords[i][0] - pp[0]
        y_temp = fid_coords[i][1] - pp[1]
        fid_coords_pp[idx][0] = x_temp
        fid_coords_pp[idx][1] = y_temp
        r_vals[idx] = math.sqrt(x_temp**2 + y_temp**2)
        idx += 1
    print(f'Principal Point Offset Correction: \n {fid_coords_pp}\n')
    print(f'Radius (r): \n {r_vals}\n')
    return fid_coords_pp, r_vals

def get_radial(coords, r, K):
    idx = 0
    rad_correction = np.zeros(shape=(len(coords),2))
    for i in range(len(coords)):
        rad_correction[idx][0] = -coords[i][0]*(K[0]*r[i]**2 + K[1]*r[i]**4 + K[2]*r[i]**6)
        rad_correction[idx][1] = -coords[i][1]*(K[0]*r[i]**2 + K[1]*r[i]**4 + K[2]*r[i]**6)
        idx += 1
    print(f'Radial Lens Distortion Correction: \n {rad_correction}\n')
    return rad_correction

def get_decentering(coords, r, P):
    idx = 0
    dec_correction = np.zeros(shape=(len(coords),2))
    for i in range(len(coords)):
        dec_correction[idx][0] = -(P[0]*(r[i]**2 + 2*coords[i][0]**2) + 2*P[1]*coords[i][0]*coords[i][1])
        dec_correction[idx][1] = -(P[1]*(r[i]**2 + 2*coords[i][1]**2) + 2*P[0]*coords[i][0]*coords[i][1])
        idx += 1
    print(f'Decentering Lens Distortion Correction: \n {dec_correction}\n')
    return dec_correction

def get_atmos(coords,r, c, K):
    idx = 0
    atmos_correction = np.zeros(shape=(len(coords),2))
    for i in range(len(coords)):
        atmos_correction[idx][0] = -coords[i][0]*K*(1 + r[i]**2/c**2)
        atmos_correction[idx][1] = -coords[i][1]*K*(1 + r[i]**2/c**2)
        idx += 1
    print(f'Atmospheric Refration Correction: \n {atmos_correction}\n')
    return atmos_correction

def new_coords(pp, rad, dec, atm):
    idx = 0
    new_coords = np.zeros(shape=(len(pp),2))
    for i in range(len(new_coords)):
        new_coords[idx][0] = pp[i][0] + rad[i][0] + dec[i][0] + atm[i][0]
        new_coords[idx][1] = pp[i][1] + rad[i][1] + dec[i][1] + atm[i][1]
        idx += 1
    print(f'Total Correction: \n {new_coords}\n')
    return new_coords

# def get_atmost:

if __name__=="__main__":
    
    # Given from calibration certificate
    focal_length = 153.358 # mm
    principal_point_offset = [-0.006, 0.006] # [xp, yp] mm
    radial_lens_distortion = [-0.8878e-4, -0.1528e-7, 0.5256e-12] # [K0, K1, K2]
    decentering_distortion = [0.1346e-06, 0.1224e-07] # [P1, P2]
    c = 3e8 # speed of light
    # Given from handout
    H = 1860 # [m] elevation
    h = 1100 # [m] ground elevation
    scale_number = 5000
    image_size = 9 # in square
    k_atmos = (2410*H)/(H**2 -6*H + 250) - (2410*h)/(h**2 - 6*h + 250)*(h/H)
    # Given from affine transformation:
    delta_X = -122.01704301790505
    delta_Y = 123.53429666924897
    A = 0.011899426266928173
    B = 2.9976774439538446e-07
    C = -1.3405013290104452e-06
    D = 0.011901264695956251
    A_mat = np.array([[A, B], [C, D]])

    xc1 = [9460, 17400, 10059, 19158, 11844, 17842, 11781, 14009, 9617, 14686]
    yc1 = [-2292, -1661, -10883, -10412, -17253, -18028, -1174, -9749, -14502, -18204]

    xc2 = [1411, 9416, 2275, 11129, 4160, 10137, 3726, 6161, 1954, 6984]
    yc2 = [-2081, -1167, -10787, -10048, -17085, -17690, -854, -9528, -14416, -17948]

    # # Image 1
    # print("Image 1:")
    # img1_fid = get_fiducial(xc1, yc1, A_mat, delta_X, delta_Y)
    # img1_pp, img1_r = get_pp(img1_fid, principal_point_offset)
    # img1_rad_correction = get_radial(img1_pp, img1_r, radial_lens_distortion)
    # img1_dec_correction = get_decentering(img1_pp, img1_r, decentering_distortion)
    # img1_atmos_correction = get_atmos(img1_pp, img1_r, c, k_atmos)
    # img1_new_coords = new_coords(img1_pp, img1_rad_correction, img1_dec_correction, img1_atmos_correction)


    # Image 2
    print('-'*75)
    print("Image 2:")    
    img2_fid = get_fiducial(xc2, yc2, A_mat, delta_X, delta_Y)
    img2_pp, img2_r = get_pp(img2_fid, principal_point_offset)
    img2_rad_correction = get_radial(img2_pp, img2_r, radial_lens_distortion)
    img2_dec_correction = get_decentering(img2_pp, img2_r, decentering_distortion)
    img2_atmos_correction = get_atmos(img2_pp, img2_r, c, k_atmos)
    img2_new_coords = new_coords(img2_pp, img2_rad_correction, img2_dec_correction, img2_atmos_correction)

