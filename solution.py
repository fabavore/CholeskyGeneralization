import numpy as np


def solve(matrix_l: np.ndarray, matrix_d: np.ndarray, vector_b: np.ndarray):
    """
    Solves the equation L D L^T x = b
    for L lower triangle matrix and D diagonal block matrix with block size 1 or 2.
    :param matrix_l: L
    :param matrix_d: D
    :param vector_b: b
    :return: x
    """

    # L, D are (n x n)-matrices and b is a (n x 1)-vector
    n = matrix_l[0,].size
    assert n == matrix_l[:, 0].size and \
        matrix_d[0, ].size == matrix_d[:, 0].size and \
        n == matrix_d[0, ].size and n == vector_b.size

    vector_z = np.zeros((n, 1))
    vector_y = np.zeros((n, 1))
    vector_x = np.zeros((n, 1))

    # Now calculate z via L z = b
    for i in range(n):
        vector_z[i] = vector_b[i] - np.dot(matrix_l[i], vector_z)

    # Calculate y via D y = z

    for i in range(n - 1):
        if matrix_d[i + 1, i] == 0:
            vector_y[i] = vector_z[i] / matrix_d[i, i]
        else:  # (2 x 2) diagonal block
            a, b, d = matrix_d[i, i], matrix_d[i + 1, i], matrix_d[i + 1, i + 1]
            if a == 0 and d == 0:
                vector_y[i] = vector_z[i] / b
                vector_y[i + 1] = vector_z[i + 1] / b
            elif a == 0:
                vector_y[i] = vector_z[i] / b
                vector_y[i + 1] = (vector_z[i + 1] - vector_z[i]) / d
            elif d == 0:
                vector_y[i + 1] = vector_z[i + 1] / b
                vector_y[i] = (vector_z[i] - vector_z[i + 1]) / a
            else:
                y = (vector_z[i] - vector_z[i + 1]) / (a * d - b * b)
                vector_y[i] = y
                vector_y[i + 1] = (vector_z[i] - a * y) / b

    # Calculate x via L^T x = y
    matrix_lt = matrix_l.T
    for i in range(n - 1, -1, -1):
        vector_x[i] = vector_y[i] - np.dot(matrix_lt[i], vector_x)

    return 0