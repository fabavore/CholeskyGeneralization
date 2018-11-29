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

A_31 = np.array([[2.25, 1, 1.5],
                 [1, 4, 2],
                 [1.5, 2, 1]])

A_32 = np.array([[1, 1, 1.5],
                 [1, 4, 2],
                 [1.5, 2, 2.25]])

A_41 = np.array([[2, 1.5, 4, 1],
                 [1.5, 0, 8, 1],
                 [4, 8, 12, 2.25],
                 [1, 1, 2.25, 0]])

P_2 = np.array([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])
L_2 = np.array([[1, 0, 0],
                [0.25, 1, 0],
                [0.5, 0.5, 1]])
D_2 = np.array([[4, 0, 0],
                [0, 2, 0],
                [0, 0, -0.5]])


def random_symmetric_matrix(n):
    """
    Generate random symmetric matrix of size n.
    """
    A = np.random.random_integers(-10, 10, size=(n, n))
    A = np.tril(A) + np.tril(A, -1).T

    return A


def test_decompose(A):
    P, L, D = dc.decompose(A)
    try:
        P, L, D = dc.decompose(A)
    except TypeError:
        print("A singular.\n")
    else:
        # print("P:\n", P, "\nL:\n", L, "\nD:\n", D, "\n")
        test_left = np.copy(A)
        for (idx_1, idx_2) in P:
            test_left[[idx_1, idx_2], :] = test_left[[idx_2, idx_1], :]
            test_left[:, [idx_1, idx_2]] = test_left[:, [idx_2, idx_1]]
        test_right = np.dot(np.dot(L, D), L.T)

        if (test_left == test_right).all():
            print("Test successful.")
        else:
            print("Left:\n", test_left, "\nRight:\n", test_right, "\n==:\n", test_left == test_right, "\n")


def random_test_decompose():
    with open('testfile', 'w') as f:
        for _ in range(1):
            test_decompose(random_symmetric_matrix(5))


def test_decompose_all():
    test_decompose(A_11)
    test_decompose(A_12)
    test_decompose(A_21)
    test_decompose(A_22)
    test_decompose(A_31)
    test_decompose(A_32)
    test_decompose(A_41)


def test_all():
    #random_test_decompose()
    test_decompose(random_symmetric_matrix(2))
    #print((np.dot(P_2.transpose(), np.dot(A_2, P_2)) ==
     #     np.dot(L_2, np.dot(D_2, L_2.transpose()))).all())


if __name__ == '__main__':
    test_all()