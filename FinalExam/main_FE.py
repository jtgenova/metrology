import numpy as np
import q1 as q1
from q2 import Question_2
import q4 as q4

def question_2():
    M = np.array([[-0.727438, -0.041167, 0.684938],
                 [0.376558, -0.858411, 0.348330],
                 [0.573619, 0.511307, 0.639943]])
    quest_2 = Question_2(M)
    quest_2.R3R2R1()
    quest_2.R1R2R3()
    quest_2.R3R1R2()


if __name__=="__main__":

    question_2()