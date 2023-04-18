import numpy as np
from numpy.linalg import inv, det
import math
from math import sin, cos
from statistics import mean, stdev

x = 106.399
# -107.16758088212826
# -19.570450942678374
# -97.96137100548914
# -9.371447257700268

w = 0.76747

xij = w + x

xp = -107.16758088212826 + xij
print(xp)

    # # Example
    # x = [106.399, 18.989, 98.681, 9.278]
    # y = [90.426, 93.365, -62.769, -92.926]

    # Xo = [7350.27, 6717.22, 6905.26, 6172.84]
    # Yo = [4382.54, 4626.41, 3279.84, 3269.45]
    # Zo = [276.42, 280.05, 266.47, 248.10]

    # c = 152.150 # mm
    # format_size = 229 # mm
    # S = 7800
    # sigma_obs = 15e-6 
    # resection = Resection(x, y, Xo, Yo, Zo, c, S, format_size, sigma_obs)
    # resection.report()

##########################################################################################################################