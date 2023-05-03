import numpy as np
from math import sin, cos
import math

class Question_2:
    def __init__(self, M):
        self.M = M
        print("-"*80)
        print("Question 2")
    
    def rot_mat(self, w, p, k):
        R1 = np.array([[1, 0 ,0],
                       [0, cos(w), sin(w)],
                       [0, -sin(w), cos(w)]])
        
        R2 = np.array([[cos(p), 0, -sin(p)],
                       [0, 1, 0],
                       [sin(p), 0, cos(p)]])
        
        R3 = np.array([[cos(k), sin(k), 0],
                       [-sin(k), cos(k), 0],
                       [0, 0, 1]])
        
        return R1, R2, R3

    def R3R2R1(self):
        w = math.atan2(-self.M[2][1], self.M[2][2])
        p = math.asin(self.M[2][0])
        k = math.atan2(-self.M[1][0], self.M[0][0])
        R1, R2, R3 = self.rot_mat(w, p, k)
        R = np.dot(R3, np.dot(R2, R1))
        print("-"*50)
        print(f"R = R3*R2*R1")
        print(f"omega = {round(math.degrees(w), 4)}")
        print(f"phi = {round(math.degrees(p), 4)}")
        print(f"kappa = {round(math.degrees(k), 4)}")

    def R1R2R3(self):
        w = math.atan2(self.M[1][2], self.M[2][2])
        p = math.asin(-self.M[0][2])
        k = math.atan2(self.M[0][1], self.M[0][0])
        R1, R2, R3 = self.rot_mat(w, p, k)
        R = np.dot(R1, np.dot(R2, R3))
        print("-"*50)
        print(f"R = R1*R2*R3")
        print(f"omega = {round(math.degrees(w), 4)}")
        print(f"phi = {round(math.degrees(p), 4)}")
        print(f"kappa = {round(math.degrees(k), 4)}")

    def R3R1R2(self):
        w = math.asin(-self.M[2][1])
        p = math.atan2(self.M[2][0], self.M[2][2])
        k = math.atan2(self.M[0][1], self.M[1][1])
        R1, R2, R3 = self.rot_mat(w, p, k)
        R = np.dot(R3, np.dot(R1, R2))
        print("-"*50)
        print(f"R = R3*R1*R2")
        print(f"omega = {round(math.degrees(w), 4)}")
        print(f"phi = {round(math.degrees(p), 4)}")
        print(f"kappa = {round(math.degrees(k), 4)}")
