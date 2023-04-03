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
import preLab4 as pre_lab_4

def rot_mat(W, P, K):
    rot_mat = np.array([
        [cos(P)*cos(K), cos(W)*sin(K)+sin(W)*sin(P)*cos(K), sin(W)*sin(K)-cos(W)*sin(P)*cos(K)],
        [-cos(P)*sin(K), cos(W)*cos(K)-sin(W)*sin(P)*sin(K), sin(W)*cos(K)+cos(W)*sin(P)*sin(K)],
        [sin(P), -sin(W)*cos(P), cos(W)*cos(P)]
    ])

    return rot_mat

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

    delta = -np.dot(np.dot(inv(np.dot(A_mat.T, A_mat)), A_mat.T), w)
    W = W + delta[0]
    P = P + delta[1]
    K = K + delta[2]
    tx = tx + delta[3]
    ty = ty + delta[4]
    tz = tz + delta[5]
    scale = scale + delta[6]

    return round(W[0], 4),  round(P[0], 4), round(K[0], 4), round(tx[0], 4), round(ty[0], 4), round(tz[0], 4), round(scale[0], 4), A_mat

def object_space(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale):
    M = rot_mat(W, P, K)
    coords = np.array([[Xm], [Ym], [Zm]])
    t_vec = np.array([tx, ty, tz])
    Xo, Yo, Zo = scale*np.dot(M, coords) + t_vec

    return round(Xo[0], 4), round(Yo[0], 4), round(Zo[0], 4)

def find_residuals(A_mat, Xg, Yg, Zg, delta):
    n = len(Xg)
    mat_size = 3*n
    l_mat = np.zeros(shape=(len(Xg)*3, 1))
    
    idx = 0
    for i in range(n):
        l_mat[idx] = Xg[i]
        l_mat[idx+1] = Yg[i]
        l_mat[idx+2] = Zg[i]
        idx += 3
    v = np.dot(A_mat, delta) - l_mat
    v_mat = np.zeros(shape=(n,3))
    idx = 0
    x_rms = 0
    y_rms = 0
    z_rms = 0

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

    print(f'vX: {v_mat[:,0]}')
    print(f'vY: {v_mat[:,1]}')
    print(f'vZ: {v_mat[:,2]}')
    print(f'x_rms: {x_rms}')
    print(f'y_rms: {y_rms}')
    print(f'z_rms: {z_rms}')
    

    
    # return x_rms, y_rms, z_rms

def resid(Xg, Yg, Zg, Xo, Yo, Zo):
    res_x = np.zeros(len(Xg))
    res_y = np.zeros(len(Yg))
    res_z = np.zeros(len(Zg))

    x_rms = 0
    y_rms = 0
    z_rms = 0

    for i in range(len(Xg)):
        res_x[i] = round((Xo[i] - Xg[i]), 4)
        res_y[i] = round((Yo[i] - Yg[i]), 4)
        res_z[i] = round((Zo[i] - Zg[i]), 4)

        x_rms = x_rms + res_x[i]**2
        y_rms = y_rms + res_y[i]**2
        z_rms = z_rms + res_z[i]**2

    x_rms = math.sqrt((1/len(Xg))*x_rms)
    y_rms = math.sqrt((1/len(Xg))*y_rms)
    z_rms = math.sqrt((1/len(Xg))*z_rms)

    return res_x, res_y, res_z, x_rms, y_rms, z_rms
    

def object_space_pc(B, W, P, K, tx, ty, tz, scale):
    M = rot_mat(W, P, K)
    t_hat = np.array([[tx], [ty], [tz]])

    rpcL = t_hat
    rpcR = scale*np.dot(M, B) + t_hat
    
    return rpcL, rpcR

def trans_angles(w, p, k, W, P, K):
    # print(math.degrees(w), math.degrees(p), math.degrees(k))
    # print(math.degrees(W), math.degrees(P), math.degrees(K))

    M_m_i_R = rot_mat(w, p, k)
    M_m_i_L = rot_mat(0, 0, 0)
    M_m_o = rot_mat(W, P, K)
    # print(f'M_m_o: \n {M_m_o}')
    M_o_i_L = np.dot(M_m_i_L, M_m_o.T)
    M_o_i_R = np.dot(M_m_i_R, M_m_o.T)

    return M_o_i_L, M_o_i_R



if __name__=="__main__":

    # Example
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
    scale = 10

    # Task #1
    iters = 10
    done = False
    for i in range(iters):
        W_old = W
        W, P, K, tx, ty, tz, scale, A_mat = find_deltas(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale, Xg, Yg, Zg)
        print(W, P, K, tx, ty, tz, scale)
        if abs(W-W_old) < 1e-4:
            print(f'Converged at {i+1} iterations!')
            break
    
    print(f'\ntx: {tx} m')
    print(f'ty: {ty} m')
    print(f'tz {tz} m')
    print(f'omega: {math.degrees(W)} deg')
    print(f'phi: {math.degrees(P)} deg')
    print(f'kappa: {math.degrees(K)} deg')
    print(f'lambda: {scale}\n')

    # print(f'A Matrix:\n {A_mat}')
    delta = np.array([W, P, K, tx, ty, tz, scale])

    # Task #2 - Convergence Criteria

    # Task #3
    Xo_vec=np.zeros(len(Xm))
    Yo_vec=np.zeros(len(Ym))
    Zo_vec=np.zeros(len(Zm))
    idx = 0
    for i in range(len(Xm)):
        Xo_vec[idx], Yo_vec[idx], Zo_vec[idx] = object_space(Xm[i], Ym[i], Zm[i], W, P, K, tx, ty, tz, scale)
        idx += 1
    print(f"Object Space X: {Xo_vec}")
    print(f"Object Space Y: {Yo_vec}")
    print(f"Object Space Z: {Zo_vec}\n")
    
    vX, vY, vZ, x_rms, y_rms, z_rms = resid(Xg, Yg, Zg, Xo_vec, Yo_vec, Zo_vec)
    print(f'vX: {vX}')
    print(f'vY: {vY}')
    print(f'vZ: {vZ}')
    print(f'x_rms: {round(x_rms, 4)}')
    print(f'y_rms: {round(y_rms, 4)}')
    print(f'z_rms: {round(z_rms, 4)}\n')

    # Task # 4
    bx = 92.000
    by = 5.0455
    bz = 2.1725
    B = np.array([[bx], [by], [bz]])
    rpcL, rpcR = object_space_pc(B, W, P, K, tx, ty, tz, scale)
    print(f'Left Image PC: \n{rpcL}')
    print(f'Right Image PC: \n{rpcR}\n')

    # Task # 5

    # Task # 7
    w = math.radians(0.4392)
    p = math.radians(1.5080)
    k = math.radians(3.1575)
    M_o_i_L, M_o_i_R = trans_angles(w, p, k, W, P, K)
    print(f'Transformation from object to image space (Left): \n {M_o_i_L}')
    print(f'Transformation from object to image space (Right): \n {M_o_i_R}')

    
    

    ###########################################################################################################################

    # # Task 1
    # Xg= [-399.28, 475.55, 517.62]
    # Yg = [-679.72, -538.18, -194.43]
    # Zg = [1090.96, 1090.5, 1090.65]
    # image_model = pre_lab_4.task_1()

    # # initial conditions
    # W = 0
    # P = 0
    # K = 0
    # tx = 0
    # ty = 0
    # tz = 0
    # scale = 10

    # iters = 3
    # iters = 3
    # for i in range(iters):
    #     W, P, K, tx, ty, tz, scale = find_deltas(Xm, Ym, Zm, W, P, K, tx, ty, tz, scale, Xg, Yg, Zg)
    # print(f'\ntx: {tx} m, ty: {ty} m, tz {tz} m, omega: {math.degrees(W)} deg, phi: {math.degrees(P)} deg, kappa: {math.degrees(K)} deg, lambda: {scale}\n')
    # delta = [W, P, K, tx, ty, tz, scale]

    # Task #2
    # find_residuals(Xm, Ym, Zm, Xg, Yg, Zg, delta)

    # # Task #3
    # Xo_vec=np.zeros(len(Xm))
    # Yo_vec=np.zeros(len(Ym))
    # Zo_vec=np.zeros(len(Zm))
    # idx = 0
    # for i in range(len(Xm)):
    #     Xo_vec[idx], Yo_vec[idx], Zo_vec[idx] = object_space(Xm[i], Ym[i], Zm[i], W, P, K, tx, ty, tz, scale)
    #     idx += 1
    # print(f"Object Space X: {Xo_vec}")
    # print(f"Object Space Y: {Yo_vec}")
    # print(f"Object Space Z: {Zo_vec}")

    # # Task # 4
    # rpcL = np.array([tx, ty, tz])

    # # Task # 5

    
