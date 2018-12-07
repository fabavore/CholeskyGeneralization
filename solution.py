import numpy as np


def forward_substitution(matrix_l, b):
    """
    Solve matrix_l * x = b via forward subsitution
    :param matrix_l: lower triangular matrix
    :param b: right side vector
    :return: solution vector x
    """
    n = matrix_l[0,].size
    x = np.zeros((n, 1))
    for i in range(n):
        x[i] = b[i] - np.dot(matrix_l[i], x)
    return x


def diagonal_substitution(matrix_d, b):
    """
    Solve matrix_d * x = b via 'diagonal substitution', see below
    :param matrix_d: block diagonal matrix with (1x1) and (2x2) blocks
    :param b: right side vector
    :return: solution vector x
    """
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
    """
    Solves matrix_u * x = b via backward substitution
    :param matrix_u: upper diagonal matrix
    :param b: right side vector
    :return: solution vector x
    """
    n = matrix_u[0,].size
    x = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(matrix_u[i], x)
    return x


def solve(matrix_l: np.ndarray, matrix_d: np.ndarray, b: np.ndarray):
    """
    Solves the equation L D L^T x = b (L = matrix_l, D = matrix_d)
    for matrix_l lower triangle matrix and matrix_d diagonal block matrix with block size 1 or 2.
    :param matrix_l: lower triangular matrix
    :param matrix_d: block diagonal matrix
    :param b: right side vector
    :return: solution vector x
    """

    # matrix_l, matrix_d are (n x n)-matrices and b is a (n x 1)-vector.
    # Assert that all matrix and vector sizes match.
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
