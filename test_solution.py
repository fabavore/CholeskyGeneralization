import numpy as np
import random_matrices
import solution as sol

eps = 1e-08


def test_solve(matrix_l, matrix_d, x):
    b = np.dot(np.dot(np.dot(matrix_l, matrix_d), matrix_l.T), x)
    x_test = sol.solve([], matrix_l, matrix_d, b)
    if not np.linalg.norm(x_test - x) < eps:
        print(np.linalg.norm(x_test - x))
        print(np.allclose(x_test, x))
        #print("Not close enough:\n", np.linalg.norm(x_test - x), "\n", x, "\n", x_test, "\n")
        return False
    return True


def test_batch(batch_size=10000, n=10, max_val=1, int_val=False):
    count = 0
    for _ in range(batch_size):
        matrix_l = random_matrices.lower_triangular(n, max_val, int_val)
        matrix_d = random_matrices.block_diagonal(n, max_val, 0.5, eps, 0.5, int_val)
        x = random_matrices.vector(n, max_val, int_val)
        count += test_solve(matrix_l, matrix_d, x)
    print("Successfully tested solution.", "\nSolved:", count)
