"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 4
Description: Resection
Deadline: April 19, 2023 10:00 AM
"""
import numpy as np
from numpy.linalg import inv, det
import math
from math import sin, cos
from statistics import mean
import preLab5 as prelab5
from statistics import mean, stdev

def find_approx(x, y, Xo, Yo, Zo, c):
    a, b, delta_x, delta_y = prelab5.similarity_transform(x, y, Xo, Yo)
    X_c = delta_x
    Y_c = delta_y
    Z_c = c*math.sqrt(a**2 + b**2) + mean(Zo)
    k = math.atan2(b, a)
    return X_c, Y_c, Z_c, k

def find_uvw(Xi, Yi, Zi, X_c, Y_c, Z_c, M):
    u = M[0][0]*(Xi - X_c) + M[0][1]*(Yi - Y_c) + M[0][2]*(Zi - Z_c)
    v = M[1][0]*(Xi - X_c) + M[1][1]*(Yi - Y_c) + M[1][2]*(Zi - Z_c)
    w = M[2][0]*(Xi - X_c) + M[2][1]*(Yi - Y_c) + M[2][2]*(Zi - Z_c)
    return u, v, w

def find_partials(Xi, Yi, Zi, X_c, Y_c, Z_c, w, p, k, xp, yp):
    M = prelab5.rot_mat(w, p, k)
    U, V, W = find_uvw(Xi, Yi, Zi, X_c, Y_c, Z_c, M)

    # dx
    dx_dXc = (-c/W**2) * (M[2][0]*U - M[0][0]*W)
    dx_dYc = (-c/W**2) * (M[2][1]*U - M[0][1]*W)
    dx_dZc = (-c/W**2) * (M[2][2]*U - M[0][2]*W)

    dx_dw = (-c/W**2)*((Yi - Y_c)*(U*M[2][2] - W*M[0][2]) - (Zi - Z_c)*(U*M[2][1] - W*M[0][1]))
    dx_dp = (-c/W**2)*((Xi - X_c)*(-W*sin(p)*cos(k) - U*cos(p)) + (Yi - Y_c)*(W*sin(w)*cos(p)*cos(k) - U*sin(w)*sin(p)) + (Zi - Z_c)*(-W*cos(w)*cos(p)*cos(k) + U*cos(w)*sin(p)))
    dx_dk = -c*V/W
    dx = [dx_dXc, dx_dYc, dx_dZc, dx_dw, dx_dp, dx_dk]

    #dy
    dy_dXc = (-c/W**2) * (M[2][0]*V - M[1][0]*W)
    dy_dYc = (-c/W**2) * (M[2][1]*V - M[1][1]*W)
    dy_dZc = (-c/W**2) * (M[2][2]*V - M[1][2]*W)

    dy_dw = (-c/W**2)*((Yi - Y_c)*(V*M[2][2] - W*M[1][2]) - (Zi - Z_c)*(V*M[2][1] - W*M[1][1]))
    dy_dp = (-c/W**2)*((Xi - X_c)*(-W*sin(p)*sin(k) - V*cos(p)) + (Yi - Y_c)*(-W*sin(w)*cos(p)*sin(k) - V*sin(w)*sin(p)) + (Zi - Z_c)*(W*cos(w)*cos(p)*sin(k) + V*cos(w)*sin(p)))
    dy_dk = c*U/W
    dy = [dy_dXc, dy_dYc, dy_dZc, dy_dw, dy_dp, dy_dk]

    x_ij = xp - c*U/W
    y_ij = yp - c*V/W

    return dx, dy, x_ij, y_ij

def find_delta(Xi, Yi, Zi, X_c, Y_c, Z_c, w, p, k, x, y, xp, yp, P):
    A_mat = np.zeros(shape=(len(Xi*2), 6))
    misclosure = np.zeros(len(Xi*2))
    idx = 0
    for i in range(len(Xi)):
        dx, dy, x_ij, y_ij = find_partials(Xi[i], Yi[i], Zi[i], X_c, Y_c, Z_c, w, p, k, xp, yp)
        A_mat[idx] = dx
        A_mat[idx+1] = dy

        misclosure[idx] = x_ij - x[i]
        misclosure[idx+1] = y_ij - y[i]
        idx += 2

    delta = -np.dot(np.dot(np.dot(inv(np.dot(np.dot(A_mat.T, P), A_mat)), A_mat.T), P), misclosure)
    X_c = X_c + delta[0]
    Y_c = Y_c + delta[1]
    Z_c = Z_c + delta[2]
    w = w + delta[3]
    p = p + delta[4]
    k = k + delta[5]

    return X_c, Y_c, Z_c, w, p, k, A_mat

def find_est(Xi, Yi, Zi, X_c, Y_c, Z_c, M, xp, yp, c):
    U, V, W = find_uvw(Xi, Yi, Zi, X_c, Y_c, Z_c, M)
    x = xp - c*U/W
    y = yp - c*V/W
    return x, y

def resid(Xi, Yi, Zi, X_c, Y_c, Z_c, w, p, k, xp, yp, c, x_hat, y_hat):
    M = prelab5.rot_mat(w, p, k)
    res_x = np.zeros(len(Xi))
    res_y = np.zeros(len(Yi))

    x_rms = 0
    y_rms = 0

    idx = 0
    for i in range(len(Xi)):
        x, y = find_est(Xi[i], Yi[i], Zi[i], X_c, Y_c, Z_c, M, xp, yp, c)
        res_x[idx] = x - x_hat[i]
        res_y[idx] = y - y_hat[i]
        x_rms = x_rms + res_x[i]**2
        y_rms = y_rms + res_y[i]**2
        idx += 1

    x_rms = math.sqrt((1/len(Xi))*x_rms)
    y_rms = math.sqrt((1/len(Xi))*y_rms)

    return res_x, res_y, x_rms, y_rms

def find_corr_matrix(A_mat):
    S_mat = np.zeros(shape=(6,6))
    C = inv(np.dot(A_mat.T, A_mat))
    for i in range(len(C)):
        S_mat[i][i] = math.sqrt(C[i][i])
    S_inv = inv(S_mat)
    R = np.dot(S_inv, np.dot(C, S_inv))

    return R

if __name__=="__main__":
####################################################################################################

    # Example
    x = [106.399, 18.989, 98.681, 9.278]
    y = [90.426, 93.365, -62.769, -92.926]

    Xo = [7350.27, 6717.22, 6905.26, 6172.84]
    Yo = [4382.54, 4626.41, 3279.84, 3269.45]
    Zo = [276.42, 280.05, 266.47, 248.10]

    c = 152.150 # mm
    format_size = 229 # mm
    sigma_obs = 15 # um
    P = 1/(sigma_obs**2)*np.identity(len(x*2))

    X_c, Y_c, Z_c, k = find_approx(x, y, Xo, Yo, Zo, c)
    arr = []
    # arr.append(X_c)
    w = 0
    p = 0

    xp = 0
    yp = 0
    for i in range(3):
        X_c, Y_c, Z_c, w, p, k, A_mat= find_delta(Xo, Yo, Zo, X_c, Y_c, Z_c, w, p, k, x, y, xp, yp, P)

    print(f'X_c: {X_c}')
    print(f'Y_c: {Y_c}')
    print(f'Z_c: {Z_c}')
    print(f'w: {math.degrees(w)}')
    print(f'phi: {math.degrees(p)}')
    print(f'kappa: {math.degrees(k)}')

    res_x, res_y, x_rms, y_rms = resid(Xo, Yo, Zo, X_c, Y_c, Z_c, w, p, k, xp, yp, c, x, y)
    print(f'Vx: {res_x}')
    print(f'Vy: {res_y}')
    print(f'x_rms: {x_rms}')
    print(f'y_rms: {y_rms}')

    print(f'Correlation Coefficient Matrix: \n{find_corr_matrix(A_mat)}')


