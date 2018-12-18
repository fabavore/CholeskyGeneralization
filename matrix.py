import numpy as np


def permute_rows(matrix_a, permutations):
    """
    Apply a number of permutations to the rows (entries) of a matrix (vector).

    :param matrix_a: matrix to permute
    :param permutations: permutations list
    :return:
    """

    for (idx_1, idx_2) in permutations:
        # Swap line idx_1 with line idx_2
        matrix_a[[idx_1, idx_2], :] = matrix_a[[idx_2, idx_1], :]


def permute_symmetric(matrix_a, permutations):
    """
    Apply a number of permutations symmetrically to the rows and columns of a matrix.

    :param matrix_a: matrix to permute
    :param permutations: permutations list
    :return:
    """

    for (idx_1, idx_2) in permutations:
        # Swap line idx_1 with line idx_2
        matrix_a[[idx_1, idx_2], :] = matrix_a[[idx_2, idx_1], :]
        # Swap column idx_1 with column idx_2
        matrix_a[:, [idx_1, idx_2]] = matrix_a[:, [idx_2, idx_1]]


def random_matrix(shape, max_val, int_val=False):
    """
    Generates a random (n x m)-matrix with int or double float entries with absolute value less than max_val.

    :param shape: matrix dimensions
    :param max_val: upper bound for absolute value
    :param int_val: if True generates int matrix
    :return: generated random matrix
    """

    return np.random.random_integers(-max_val, max_val, shape) if int_val \
        else 2 * (np.random.random_sample(shape) - 0.5 * np.ones(shape)) * max_val


def random_symmetric_matrix(n, max_val, int_val=False):
    """
    Generates a random symmetric (n x n)-matrix with int or double float entries with absolute value less than max_val.

    :param n: matrix dimension
    :param max_val: upper bound for absolute value
    :param int_val: if True generates int matrix
    :return: generated random symmetric matrix
    """

    a = random_matrix((n, n), max_val, int_val)
    return np.tril(a) + np.tril(a, -1).T
