import numpy as np
import matrix


eps = np.finfo(float).eps
epsilon = 0.5


def get_diagonal_indices(matrix_a: np.ndarray):
    """
    Get the position of the biggest diagonal or biggest non-diagonal item
    (according to given rule set) as a list containing 1 or 2 indices.

    :param matrix_a: matrix A
    :return: index list, None if A is singular
    """

    # A is a (n x n)-matrix
    n = matrix_a[0, ].size

    xi_diag = 0
    k = -1
    for i in range(n):
        val = abs(matrix_a[i, i])
        if val > xi_diag:
            xi_diag = val
            k = i

    xi_off = 0
    l, m = -1, -1
    for i in range(1, n):
        for j in range(i):
            val = abs(matrix_a[i, j])
            if val > xi_off:
                xi_off = val
                l, m = i, j

    if xi_diag < eps and xi_off < eps:
        return None
    elif xi_diag != 0 and xi_off / xi_diag <= 1 + epsilon:
        return [k]
    else:
        return [m, l]


def decompose_recursive(idx, matrix_a: np.ndarray, permutations: list, matrix_l: np.ndarray, matrix_d: np.ndarray):
    """
    Perform the idx-th recursion step of the decomposition for remaining matrix_a,
    i.e. calculate the idx-th diagonal block of diagonal matrix matrix_d,
    the idx-th column of the lower triangular matrix matrix_l and the idx-th permutation.

    :param idx: recursion index
    :param matrix_a: remaining matrix to be decomposed
    :param permutations: permutations list to append current permutation to
    :param matrix_l: lower triangular matrix to write current column to
    :param matrix_d: block diagonal matrix to write current diagonal block to
    :return: True if matrix_a can be decomposed, False if it is singular
    """

    # matrix_a is (n x n)-matrix
    n = matrix_a[0, ].size

    # First, check if matrix_a is singular, if not so determine the indices and size of the diagonal block.
    diag_indices = get_diagonal_indices(matrix_a)
    if diag_indices is None:
        # Matrix is singular, so it cannot be decomposed.
        return False
    else:
        # Size of the diagonal block.
        diag_size = len(diag_indices)
        assert diag_size in [1, 2]

    # ----- Recursion termination ----- #

    if n == 1:
        # Decomposition complete.
        matrix_d[idx, idx] = matrix_a[0, 0]
        return True

    elif n == 2:
        # Decomposition complete.
        if diag_size == 1:
            # Just some simple calculations...
            diag_idx = diag_indices[0]
            d = matrix_a[diag_idx, diag_idx]
            c = matrix_a[0, 1]
            h = matrix_a[1 - diag_idx, 1 - diag_idx]

            matrix_d[idx, idx] = d
            matrix_l[idx + 1, idx] = c / d

            # Write permutation to permutations list if necessary
            if diag_idx == 1:
                matrix.permute_rows(matrix_l[idx:, 0:idx], [(0, 1)])
                permutations.append((idx, idx + 1))

            # Perform last recursion step and check if the matrix is singular.
            return decompose_recursive(idx + 1, np.array([[h - c * c / d]]), permutations, matrix_l, matrix_d)
        else:  # diag_size == 2
            matrix_d[idx:idx + 2, idx:idx + 2] = matrix_a[0:2, 0:2]
            return True

    # -------- Recursion step --------- #

    else:
        # Permute matrix_a symmetric and in-place to get the a block matrix with diagonal block as upper left block
        # and save the permutation indices for further calculations.
        current_permutations = []
        for i in range(diag_size):  # either 1 or 2
            current_permutations.append((i, diag_indices[i]))

        matrix.permute_symmetric(matrix_a, current_permutations)
        matrix.permute_rows(matrix_l[idx:, 0:idx], current_permutations)

        for (i, j) in current_permutations:
            permutations.append((i + idx, j + idx))

        # Calculate the inverse of the diagonal block for further calculations.
        if diag_size == 1:
            diag_det = matrix_a[0, 0]
            diag_matrix = np.identity(1)
        else:  # diag_size == 2
            a, b, d = matrix_a[0, 0], matrix_a[0, 1], matrix_a[1, 1]
            diag_det = a * d - b * b
            diag_matrix = np.array([[d, -b], [-b, a]])

        # Write the diagonal block into the block diagonal matrix.
        matrix_d[idx:idx + diag_size, idx:idx + diag_size] = matrix_a[0:diag_size, 0:diag_size]

        # Calculate matrix_l column and matrix_a_new for the next recursion step
        matrix_c = matrix_a[diag_size:, 0:diag_size]
        if diag_size == 1:
            matrix_l[idx + diag_size:, idx:idx + diag_size] = matrix_c / diag_det
            matrix_a_new = matrix_a[diag_size:, diag_size:] - np.dot(matrix_c, matrix_c.T) / diag_det
        else:  # diag_size == 2
            matrix_l[idx + diag_size:, idx:idx + diag_size] = np.dot(matrix_c, diag_matrix) / diag_det
            matrix_a_new = matrix_a[diag_size:, diag_size:] - \
                np.dot(np.dot(matrix_c, diag_matrix), matrix_c.T) / diag_det

        return decompose_recursive(idx + diag_size, matrix_a_new, permutations, matrix_l, matrix_d)


def decompose(matrix_a: np.ndarray):
    """
    If matrix_a is not singular, decompose matrix_a with the generalized Cholesky decomposition
    such that P^T A P = L D L^T (P from permutation list, A = matrix_a, L = matrix_l, D = matrix_d).

    :param matrix_a: matrix to be decomposed
    :return: matrices matrix_l, matrix_d and permutation list, None if matrix_a is singular
    """

    # make sure that matrix_a is symmetric
    assert (matrix_a == matrix_a.T).all()
    # matrix_a is a (n x n)-matrix
    n = matrix_a[0, ].size

    # Copy matrix_a because it will be changed in-place
    matrix_a_copy = np.copy(matrix_a)
    # Permutation list
    permutations = []
    # Lower triangular matrix
    matrix_l = np.identity(n)
    # Diagonal matrix
    matrix_d = np.identity(n)

    if decompose_recursive(0, matrix_a_copy, permutations, matrix_l, matrix_d):
        return permutations, matrix_l, matrix_d
