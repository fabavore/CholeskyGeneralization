import numpy as np
import random_matrices
import decomposition as dc

eps = 1e-14


def test_decompose(matrix_a):
    try:
        permutations, matrix_l, matrix_d = dc.decompose(matrix_a)
    except TypeError:
        if np.linalg.det(matrix_a) < eps:
            return False
        else:
            print("Something went wrong:\n", np.linalg.det(matrix_a), "\n", matrix_a)
            raise TypeError

    dc.permute_symmetric(matrix_a, permutations)
    matrix_a_test = np.dot(np.dot(matrix_l, matrix_d), matrix_l.T)
    if not np.linalg.norm(matrix_a_test - matrix_a) < eps:
        print("Not close enough:\n", np.linalg.norm(matrix_a_test - matrix_a), "\n", matrix_a, "\n", matrix_a_test, "\n")
        raise TypeError
    return True


def test_batch(batch_size=10000, n=10, max_val=1, int_val=False):
    count = 0
    for _ in range(batch_size):
        count += test_decompose(random_matrices.symmetric(n, max_val, int_val))
    print("Successfully tested decomposition.", "\nDecomposed:", count, "\nSingular matrices:", batch_size - count)
