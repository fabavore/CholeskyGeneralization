import numpy as np
import decomposition as dc

A_11 = np.array([[0]])

A_12 = np.array([[2]])

A_21 = np.array([[2, 1],
                 [1, 0]])

A_22 = np.array([[0, 1],
                 [1, 2]])

A_23 = np.array([[0.5, 2],
                 [2, 0]])

A_3 = np.array([[   2.25   , 1     , 1.5   ],
                [   1      , 4     , 2     ],
                [   1.5    , 2     , 1     ]])

P_2 = np.array([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])
L_2 = np.array([[1, 0, 0],
                [0.25, 1, 0],
                [0.5, 0.5, 1]])
D_2 = np.array([[4, 0, 0],
                [0, 2, 0],
                [0, 0, -0.5]])


def test_decompose(A):
    try:
        P, L, D = dc.decompose(A)
    except TypeError:
        print("A singular.\n")
    else:
        print("P:\n", P, "\nL:\n", L, "\nD:\n", D, "\n")


def test_decompose_all():
    test_decompose(A_11)
    test_decompose(A_12)
    test_decompose(A_21)
    test_decompose(A_22)
    test_decompose(A_3)


def test_all():
    test_decompose_all()
    #print((np.dot(P_2.transpose(), np.dot(A_2, P_2)) ==
     #     np.dot(L_2, np.dot(D_2, L_2.transpose()))).all())


if __name__ == '__main__':
    test_all()