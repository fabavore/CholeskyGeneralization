import test_decomposition
import test_solution

import numpy as np
import random_matrices
import decomposition
import solution


def test_decompose_and_solve():
    matrix_a = random_matrices.symmetric(10, 1)
    assert np.linalg.det(matrix_a) > 0
    x = random_matrices.vector(10, 1)
    b = np.dot(matrix_a, x)
    permutations, matrix_l, matrix_d = decomposition.decompose(matrix_a)
    x_test = solution.solve(permutations, matrix_l, matrix_d, b)

    print(x, "\n\n", x_test, "\n\n", np.allclose(x, x_test))


if __name__ == '__main__':
    #test_decomposition.test_batch()
    #test_solution.test_batch()
    test_decompose_and_solve()
