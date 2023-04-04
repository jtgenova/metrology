import numpy as np
import math

# Age = [10, 12, 20, 10, 8]
# Rating = [3, 4, 10, 1, 7]
# MMPI = [38, 34, 74, 40, 64]

# A_mat = np.array([Age, Rating, MMPI]).T
# N = len(Age)
# print(f'A matrix:\n {A_mat}')

# Age_mean = sum(Age)/len(Age)
# Rating_mean = sum(Rating)/len(Rating)
# MMPI_mean = sum(MMPI)/len(MMPI)
# x_bar = np.array([[Age_mean], [Rating_mean], [MMPI_mean]])


# D_mat = np.zeros(shape=(len(Age), 3))
# print(D_mat)
# idx = 0
# for i in range(len(Age)):
#     D_mat[idx][0] = Age[i] - Age_mean
#     D_mat[idx][1] = Rating[i] - Rating_mean
#     D_mat[idx][2] = MMPI[i] - MMPI_mean
#     idx += 1

A1 = [13.693, 13.250, 13.737, 13.370, 13.202, 13.835]
A2 = [7.4894, 8.0166, 0.0901, -0.6357, -8.0956, -6.1057]
A3 = [2818.2, 2831.7, 2146.8, 2106.7, 2937.6, 2594.7]
A4 = [22.15, 719.55, 117.77, 44.07, -590.12, 148.96]
A5 = [386.93, -807.64, -220.84, -1196.20, -1059.30, 142.19]

A_mat = np.array([A1, A2, A3, A4, A5]).T
N = len(A1)
print(f'A matrix:\n {A_mat}')

A1_mean = sum(A1)/len(A1)
A2_mean = sum(A2)/len(A2)
A3_mean = sum(A3)/len(A3)
A4_mean = sum(A4)/len(A4)
A5_mean = sum(A5)/len(A5)

x_bar = np.array([[A1], [A2], [A3], [A4], [A5]])


D_mat = np.zeros(shape=(len(A1), 5))
print(D_mat)
idx = 0
for i in range(len(A1)):
    D_mat[idx][0] = A1[i] - A1_mean
    D_mat[idx][1] = A2[i] - A2_mean
    D_mat[idx][2] = A3[i] - A3_mean
    D_mat[idx][3] = A4[i] - A4_mean
    D_mat[idx][4] = A5[i] - A5_mean

    idx += 1

print(f'D matrix:\n {D_mat}')

CCSP = np.dot(D_mat.T, D_mat)
print(f'CCSP matrix:\n {CCSP}')

C = CCSP*(1/(N-1))
print(f'C Matrix: \n {C}')

S = C.diagonal()
S_mat = np.zeros(shape=(len(C), len(C)))
idx = 0
for i in range(len(C)):
    S_mat[idx][idx] = math.sqrt(S[i])
    idx += 1
print(f'S Matrix: \n {S_mat}')

S_inv = np.linalg.inv(S_mat)
R = np.dot(np.dot(S_inv, C), S_inv)
print(f'R Matrix: \n {R}')


