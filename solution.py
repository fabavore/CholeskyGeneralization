import numpy as np


def forward_substitution(matrix_l, b):
    n = matrix_l[0,].size
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = b[i] - np.dot(matrix_l[i], x)
    return x


def diagonal_substitution(matrix_d, b):
    n = matrix_d[0,].size
    x = np.zeros((n, 1))
    i = 0
    while i < n - 1:
        if matrix_d[i + 1, i] == 0:  # 1x1 diagonal block, easy
            x[i] = b[i] / matrix_d[i, i]
            i += 1
        else:  # 2x2 diagonal block
            a, c, d = matrix_d[i, i], matrix_d[i + 1, i], matrix_d[i + 1, i + 1]
            b_1, b_2 = b[i, 0], b[i + 1, 0]
            # b != 0 for non-singular matrix!
            if a == 0 and d == 0:
                x_1 = b_2 / c
                x_2 = b_1 / c
            elif a == 0:
                x_2 = b_1 / c
                x_1 = (b_2 - d * x_2) / c
            elif d == 0:
                x_1 = b_2 / c
                x_2 = (b_1 - a * x_1) / c
            else:
                x_1 = (d * b_1 - c * b_2) / (a * d - c * c)
                x_2 = (b_1 - a * x_1) / c
            x[i] = x_1
            x[i + 1] = x_2
            i += 2
    if i < n:
        x[i] = b[i] / matrix_d[i, i]
    return x


def backward_substitution(matrix_u, b):
    n = matrix_u[0,].size
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(matrix_u[i], x)
    return x


def solve(matrix_l: np.ndarray, matrix_d: np.ndarray, b: np.ndarray):
    """
    Solves the equation L D L^T x = b
    for L lower triangle matrix and D diagonal block matrix with block size 1 or 2.
    :param matrix_l: L
    :param matrix_d: D
    :param vector_b: b
    :return: x
    """

    # L, D are (n x n)-matrices and b is a (n x 1)-vector
    n = matrix_l[0, ].size
    assert n == matrix_l[:, 0].size and \
        matrix_d[0, ].size == matrix_d[:, 0].size and \
        n == matrix_d[0, ].size and n == b.size

    # Now calculate z from L z = b via forward substitution
    z = forward_substitution(matrix_l, b)

    # Calculate y from D y = z via diagonal substitution
    y = diagonal_substitution(matrix_d, z)

    # Calculate x from L^T x = y via backward substitution
    x = backward_substitution(matrix_l.T, y)

    return x
