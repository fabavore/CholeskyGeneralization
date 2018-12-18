from typing import Any, Union

import numpy as np
import sys

import matrix
import decomposition
import solution


precision = 1e-12


def test_run(matrix_a, x):
    """
        Perform a single test run for decompose and solve.

        :param matrix_a: Non-singular matrix
        :param x:
        :return:
        """

    # Calculate right side vector.
    b = np.dot(matrix_a, x)

    # Calculate decomposition, return if matrix is singular.
    try:
        permutations, matrix_l, matrix_d = decomposition.decompose(matrix_a)
    except TypeError:
        if np.linalg.det(matrix_a) < decomposition.eps:
            print("Unrecognized singular matrix:\n", matrix_a)
        else:
            print("Unknown Error: {0}\n".format(np.linalg.det(matrix_a)), matrix_a)
        return False

    matrix.permute_symmetric(matrix_a, permutations)
    if not np.linalg.norm(matrix_a - np.dot(np.dot(matrix_l, matrix_d), matrix_l.T)) < precision:
        # Decomposition does not meet precision requirement
        print("Not close enough in decomposition: {0}\n".format(
            np.linalg.norm(matrix_a - np.dot(np.dot(matrix_l, matrix_d), matrix_l.T))
        ), matrix_a)
        return False

    if not np.linalg.norm(x - solution.solve(permutations, matrix_l, matrix_d, b)) < precision:
        # Equation system solution does not meet precision requirement
        print("Not close enough in solution: {0}, {1}\n".format(
            np.linalg.norm(x - solution.solve(permutations, matrix_l, matrix_d, b)),
            np.linalg.norm(matrix_a)
        ), matrix_a, "\n", x)
        return False

    return True


def random_test_run(n, max_val, int_val):
    matrix_a = matrix.random_symmetric_matrix(n, max_val, int_val)
    while np.linalg.det(matrix_a) < 1e-8:
        matrix_a = matrix.random_symmetric_matrix(n, max_val, int_val)
    x = matrix.random_matrix((n, 1), max_val, int_val)
    return test_run(matrix_a, x)


def random_test_batch(size, n, max_val, int_val):
    print("Testing {0} ({1} x {1})-matrices with {3} values in [-{2}, {2}{4}\nStarting test...".format(
        size, n, max_val, "integer" if int_val else "float", "]" if int_val else ")"
    ))
    cnt = 0
    with open("error_log", "a") as sys.stdout:
        for _ in range(size):
            cnt += random_test_run(n, max_val, int_val)
    sys.stdout = sys.__stdout__
    print("Test complete and successful for {0} random matrices.".format(cnt))


if __name__ == '__main__':
    random_test_batch(10000, 4, 4, True)
