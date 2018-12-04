import numpy as np
import decomposition as dc


def random_symmetric_int_matrix(n, max_val):
    """
    Generate random symmetric matrix of size n.
    """
    A = np.random.random_integers(-max_val, max_val, size=(n, n))
    A = np.tril(A) + np.tril(A, -1).T
    return A


def random_symmetric_positive_int_matrix(n, max_val):
    """
        Generate random symmetric matrix of size n with positive entries.
        """
    A = np.random.random_integers(0, max_val, size=(n, n))
    A = np.tril(A) + np.tril(A, -1).T
    return A


def test_decompose(matrix_a):
    try:
        permutations, matrix_l, matrix_d = dc.decompose(matrix_a)
    except TypeError:
        print("Singular matrix:\n", matrix_a)
        return False

    matrix_a_copy = np.copy(matrix_a)

    dc.permute_symmetric(matrix_a, permutations)

    probe = np.dot(np.dot(matrix_l, matrix_d), matrix_l.T)

    if np.array_equal(matrix_a, probe):
        print("Success.\n")
    elif np.allclose(matrix_a, probe, ):
        print("Close enough.\n")
        if not (probe == probe.T).all():
            print("A:\n", matrix_a,
                "\nProbe A:\n", probe)
                # "\nGleich?:\n", A == probe, "\n")
    else:
        print("\n --- W R O N G !!! ---\n")
        print("P:\n", permutations, "\nL:\n", matrix_l, "\nD:\n", matrix_d, "\n")
        print("A:\n", matrix_a_copy,
              "\nA permuted:\n", matrix_a,
              "\nProbe A:\n", probe,
              "\nGleich?:\n", matrix_a == probe, "\n")
    return True


def random_test_decompose(n):
    assert(test_decompose(random_symmetric_int_matrix(n, 5)))


A_1 = np.array([[2, -1, -2, 4],
                [-1, 1, 1, -1],
                [-2, 1, 1, -1],
                [4, -1, -1, 2]])
A_2 = np.array([[1, 2, 1, 0],
                [2, 2, 2, 1],
                [1, 2, 0, 0],
                [0, 1, 0, 2]])
A_3 = np.array([[1, 2, 0, -1],
                [2, -2, -1, 1],
                [0, -1, 1, -2],
                [-1, 1, -2, 2]])
A_4 = np.array([[2, -1, -2, 1],
                [-1, 1, 4, -1],
                [-2, 4, 1, -1],
                [1, -1, -1, 2]])
A_5 = np.array([[4, 0, 5, 1],
                [0, 1, 0, 5],
                [5, 0, 5, 3],
                [1, 5, 3, 2]])


if __name__ == '__main__':
    #for _ in range(5):
     #   random_test_decompose(4)
     test_decompose(A_5)
