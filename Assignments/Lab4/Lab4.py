"""
CIVE 6374 – Optical Imaging Metrology
Professor: Dr. Craig Glennie
Author: Joshua Genova
Lab # 4
Description: Absolute Orientation
Deadline: April 05, 2023 10:00 AM
"""
import numpy as np
from numpy.linalg import inv, det
import math
from math import sin, cos
import matplotlib.pyplot as plt

def rot_mat(W, P, K):
    rot_mat = np.array([
        [cos(P)*cos(K), cos(W)*sin(K)+sin(W)*sin(P)*cos(K), sin(W)*sin(K)-cos(W)*sin(P)*cos(K)],
        [-cos(P)*sin(K), cos(W)*cos(K)-sin(W)*sin(P)*sin(K), sin(W)*cos(K)+cos(W)*sin(P)*sin(K)],
        [sin(P), -sin(W)*cos(P), cos(W)*cos(P)]
    ])

    return rot_mat

def object_space(scale, M, Xm, Ym, Zm, tx, ty, tz):
    M_vec = np.array([[Xm], [Ym], [Zm]])
    t_vec = np.array([[tx], [ty], [tz]])
    Xo, Yo, Zo = scale*np.dot(M, M_vec) + t_vec
    return Xo, Yo, Zo

def partial_d(Xm, Ym, Zm, W, P, K, scale, M):
    sW = sin(W)
    cW = cos(W)
    sP = sin(P)
    cP = cos(P)
    sK = sin(K)
    cK = cos(K)

    m11 = M[0][0]
    m12 = M[0][1]
    m13 = M[0][2]
    m21 = M[1][0]
    m22 = M[1][1]
    m23 = M[1][2]
    m31 = M[2][0]
    m32 = M[2][1]
    m33 = M[2][2]

    # partial derivatives of X
    dXdW = scale*Ym*(-sW*sK + cW*sP*cK) + scale*Zm*(cW*sK + sW*sP*cK)
    dXdP = -scale*Xm*sW*cK + scale*Ym*sW*cP*cK - scale*Zm*cW*cP*cK
    dXdK = -scale*Xm*cP*sK + scale*Ym*(cW*cK - sW*sP*sK) + scale*Zm*(sW*cK + cW*sP*sK)
    dXdscale = Xm*m11 + Ym*m12 + Zm*m13
    dXdtx = 1
    dXdty = 0
    dXdtz = 0
    dX = [dXdW, dXdP, dXdK, dXdtx, dXdty, dXdtz, dXdscale]
    # print(dX)

    # partial derivatives of Y
    dYdW = scale*Ym*(-sW*cK - sW*sP*sK) + scale*Zm*(cW*cK - sW*sP*sK)
    dYdP = scale*Xm*sP*sK - scale*Ym*sW*cP*sK + scale*Zm*cW*cP*sK
    dYdK = -scale*Xm*cP*cK + scale*Ym*(-cW*sK - sW*sP*cK) + scale*Zm*(-sW*sK + cW*sP*cK)
    dYdscale = Xm*m21 + Ym*m22 + Zm*m23
    dYdtx = 0
    dYdty = 1
    dYdtz = 0
    dY = [dYdW, dYdP, dYdK, dYdtx, dYdty, dYdtz, dYdscale]
    # print(dY)

    # partial derivatives of Z
    dZdW = -scale*Ym*cW*cP - scale*Zm*sW*cP
    dZdP = scale*Xm*cP + scale*Ym*sW*sP - scale*Zm*cW*sP
    dZdK = 0
    dZdscale = Xm*m31 + Ym*m32 + Zm*m33
    dZdtx = 0
    dZdty = 0
    dZdtz = 1
    dZ = [dZdW, dZdP, dZdK, dZdtx, dZdty, dZdtz, dZdscale]
    # print(dZ)

    return dX, dY, dZ

def misclosure(Xm, Ym, Zm, tx, ty, tz, scale, M, Xg, Yg, Zg):
    m11 = M[0][0]
    m12 = M[0][1]
    m13 = M[0][2]
    m21 = M[1][0]
    m22 = M[1][1]
    m23 = M[1][2]
    m31 = M[2][0]
    m32 = M[2][1]
    m33 = M[2][2]

    wX = scale*(m11*Xm + m12*Ym + m13*Zm) + tx - Xg
    wY = scale*(m21*Xm + m22*Ym + m23*Zm) + ty - Yg
    wZ = scale*(m31*Xm + m32*Ym + m33*Zm) + tz - Zg


    return wX, wY, wZ

def find_deltas(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale, Xg, Yg, Zg):
    M = rot_mat(W, P, K)
    size = len(Xm)
    A_mat = np.zeros(shape=(3*size, 7))
    w = np.zeros(shape=(3*size, 1))
    idx = 0
    for i in range(size):
        A_mat[idx], A_mat[idx+1], A_mat[idx+2] = partial_d(Xm[i], Ym[i], Zm[i], W, P, K, scale, M)
        w[idx], w[idx+1], w[idx+2] = misclosure(Xm[i], Ym[i], Zm[i], tx, ty, tz, scale, M, Xg[i], Yg[i], Zg[i])
        idx += 3

    # print(inv(np.dot(A_mat.T, A_mat)))
    # print(w)
    delta = -np.dot(np.dot(inv(np.dot(A_mat.T, A_mat)), A_mat.T), w)
    W = W + delta[0]
    P = P + delta[1]
    K = K + delta[2]
    tx = tx + delta[3]
    ty = ty + delta[4]
    tz = tz + delta[5]
    scale = scale + delta[6]

    return W, P, K, tx, ty, tz, scale

def find_residuals(A_mat, delta, l_mat):
    n = len(l_mat)
    mat_size = 2*n
    v = np.dot(A_mat, delta) - l_mat
    v_mat = np.zeros(shape=(n,3))
    idx = 0
    x_rms = 0
    y_rms = 0

    # calculate rms
    for i in range(0, mat_size, 3):
        v_mat[idx][0] = v[i]
        v_mat[idx][1] = v[i+1]
        v_mat[idx][2] = v[i+2]
        x_rms = x_rms + v[i]**2
        y_rms = y_rms + v[i+1]**2
        z_rms = z_rms + v[i+2]**2
        idx += 1
    x_rms = math.sqrt((1/n)*x_rms)
    y_rms = math.sqrt((1/n)*y_rms)
    z_rms = math.sqrt((1/n)*z_rms)
    
    return x_rms, y_rms, z_rms

def object_space_coords(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale):
    M = rot_mat(W, P, K)
    t_hat = np.array([[tx], [ty], [tz]])
    B = np.array([[Xm], [Ym], [Zm]])

    rpcL = t_hat
    rpcR = scale*np.dot(M, B) + t_hat
    
    return rpcL, rpcR


if __name__=="__main__":
    Xm = [108.9302, 19.5304, 71.8751, -0.9473, 9.6380, 100.4898]
    Ym = [92.5787, 96.0258, 4.9657, -7.4078, -96.5329, -63.9177]
    Zm = [-155.7696, -156.4878, -154.1035, -154.8060, -158.0535, -154.9389]

    Xg = [7350.27, 6717.22, 6869.09, 6316.06, 6172.84, 6905.26]
    Yg = [4382.54, 4626.41, 3844.56, 3934.63, 3269.45, 3279.84]
    Zg = [276.42, 280.05, 283.11, 283.03, 248.10, 266.47]


    # initial conditions
    W = 0
    P = 0
    K = 0
    tx = 0
    ty = 0
    tz = 0
    scale = 1

    iters = 9
    for i in range(iters):
        W, P, K, tx, ty, tz, scale = find_deltas(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale, Xg, Yg, Zg)
        print(tx, ty, tz, math.degrees(W), math.degrees(P), math.degrees(K), scale)


