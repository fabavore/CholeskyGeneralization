import numpy as np

eps = np.finfo(float).eps
epsilon = 0.5


def get_diagonal_index_list(A):
    """
    Get the position of the biggest diagonal or biggest non-diagonal item
    (according to given rule set) as a list containing 1 or 2 indices.

    :param A: matrix A
    :return: index list
    """

    # A is a (n x n)-matrix
    n = A[0, ].size

    xi_diag = 0
    k = -1
    for i in range(n):
        val = abs(A[i, i])
        if val > xi_diag:
            xi_diag = val
            k = i

    xi_off = 0
    l, m = -1, -1
    for i in range(1, n):
        for j in range(i):
            val = abs(A[i, j])
            if val > xi_off:
                xi_off = val
                l, m = i, j

    if xi_diag < eps and xi_off < eps:
        return None
    elif xi_diag != 0 and xi_off / xi_diag <= 1 + epsilon:
        return [k]
    else:
        return [l, m]


def ggg():
    return