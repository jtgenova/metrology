import numpy as np
import math

class Question_3:
    def __init__(self, xc, yc, xf, yf, xp, yp):
        self.xc = xc
        self.yc = yc
        self.xf = xf
        self.yf = yf
        self.xp = xp
        self.yp = yp
        print('-'*80)
        print("Question 3")

    def affine_transform(self):

        print('-'*50)
        print('Affine Transform')
        n = len(self.xc)
        mat_size = 2*n

        # create l-vector
        l_mat = np.zeros(shape=(mat_size,1))
        idx = 0
        for i in range(0, mat_size, 2):
            l_mat[i] = self.xf[idx]
            l_mat[i+1] = self.yf[idx]
            idx += 1

        # create A-matrix
        A_mat = np.zeros(shape=(mat_size,6))
        idx = 0
        for i in range(0, mat_size, 2):
            A_mat[i] = [self.xc[idx], self.yc[idx], 1, 0, 0 ,0]
            A_mat[i+1] = [0, 0, 0, self.xc[idx], self.yc[idx], 1]
            idx +=1 

        # calculate the unknowns x_hat
        x_hat = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A_mat),A_mat)), np.transpose(A_mat)), l_mat)
        A, B, Dx, C, D, Dy = float(x_hat[0]), float(x_hat[1]), float(x_hat[2]), float(x_hat[3]), float(x_hat[4]), float(x_hat[5])
        theta = math.atan(C/A)
        Sx = math.sqrt(A**2 + C**2)
        Sy = math.sqrt(B**2 + D**2)
        delta =math.atan((A*B + C*D)/(A*D - B*C))

        self.xf, self.yf = np.dot(np.array([[A, B], [C, D]]), np.array([self.xp, self.yp])) + np.array([Dx, Dy])
        print(f"xf: {round(self.xf, 4)}, yf: {round(self.yf, 4)}")

        # create residuals vector
        v = np.dot(A_mat, x_hat) - l_mat
        v_mat = np.zeros(shape=(n,2))
        idx = 0
        x_rms = 0
        y_rms = 0

        # calculate rms
        for i in range(0, mat_size, 2):
            v_mat[idx][0] = v[i]
            v_mat[idx][1] = v[i+1]
            x_rms = x_rms + v[i]**2
            y_rms = y_rms + v[i+1]**2
            idx += 1
        x_rms = math.sqrt((1/n)*x_rms)
        y_rms = math.sqrt((1/n)*y_rms)
        print('-'*50)
        # print('Residuals: ')
        # print(v_mat)
        print(f'x RMS {x_rms}')
        print(f'y RMS {y_rms}')

        return 
