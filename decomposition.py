import numpy as np

eps = np.finfo(float).eps
epsilon = 0.5


def get_diagonal_index_list(A: np.ndarray):
    """
    Get the position of the biggest diagonal or biggest non-diagonal item
    (according to given rule set) as a list containing 1 or 2 indices.

    :param A: matrix A
    :return: index list, None if A is singular
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


def decompose_recursive(idx, A: np.ndarray, P, L: np.ndarray, D: np.ndarray, A_orig):
    """
    Helper method for decompose(A), performs the idx-th recursion step.

    :param idx: recursion index
    :param A: matrix to be decomposed
    :param P: permutation list
    :param L: lower triangular matrix
    :param D: block diagonal matrix
    :return: diagonal indices of the current recursion step, None if A is singular
    """

    # A is (n x n)-matrix
    n = A[0,].size

    # First, check if A is singular, if not so determine the indices and size of the diagonal block.
    diag_indices = get_diagonal_index_list(A)

    if diag_indices is None:
        # Matrix is singular, so it cannot be decomposed.
        return None
    else:
        # Size of the diagonal block.
        diag_size = len(diag_indices)
        assert diag_size in [1, 2]

    # ----- Recursion termination ----- #

    if n == 1:
        # Deomposition complete.
        D[idx, idx] = A[0, 0]
        return [0]

    elif n == 2:
        # Decomposition complete.
        if diag_size == 1:
            # Just some simple calculations...
            diag_idx = diag_indices[0]
            d = A[diag_idx, diag_idx]
            c = A[0, 1]
            h = A[1 - diag_idx, 1 - diag_idx]
            D[idx:idx + 2, idx:idx + 2] = np.array([[d, 0],
                                                    [0, h - c * c / d]])
            L[idx + 1, idx] = c / d

            # Write permutation to P, then return index for previous recursion step.
            permutation_idx = idx + diag_idx
            P.append((idx, permutation_idx))

            return [diag_idx]
        else:  # diag_size == 2
            D[idx:idx + 2, idx:idx + 2] = A[0:2, 0:2]
            return [0]

    # -------- Recursion step --------- #

    else:
        # Permute A symmetric and in-place to get the desired block matrix with D_idx as upper left block
        # and add the permutation indices to the list P
        for i, swap_idx in enumerate(diag_indices):
            # Swap line i with line diag_idx
            A[[i, swap_idx], :] = A[[swap_idx, i], :]
            # Swap column i with column diag_idx
            A[:, [i, swap_idx]] = A[:, [swap_idx, i]]

        # Write permutation to P.
        permutation_idx = idx + diag_idx
        P.append((idx, permutation_idx))

        # Write the diagonal block into the block diagonal matrix.
        D[idx:idx + diag_size, idx:idx + diag_size] = A[0:diag_size, 0:diag_size]

        # Calculate the inverse of the diagonal block for further calculations.
        if diag_size == 1:
            D_inv = np.array([[1 / A[0, 0]]])
        else:  # diag_size == 2
            a, b, d = A[0, 0], A[0, 1], A[1, 1]
            D_inv = np.array([[a, -b], [-b, d]]) / (a * d - b * b)

        # Select C for calulation of the next A and L column
        C = A[diag_size:, 0:diag_size]

        # Set A for next recursion step and calculate next recursion step
        A = A[diag_size:, diag_size:] - np.dot(np.dot(C, D_inv), C.T)
        next_diag_indices = decompose_recursive(idx + diag_size, A, P, L, D, A_orig)

        # Calculate the L column(s) and permute rows accordingly
        L_col = np.dot(C, D_inv)
        for i, diag_idx in enumerate(next_diag_indices):
            L_col[[i, diag_idx], :] = L_col[[diag_idx, i], :]

        # Set L column(s)
        L[idx + diag_size:, idx:idx + diag_size] = L_col

        return diag_indices


def decompose(A: np.ndarray):
    """
    If A is not singular, decompose the given matrix
    with the generalized Cholesky decomposition
    such that P^T A P = L D L^T.

    :param A: matrix A
    :return: matrices P, L, D, None if A is singular
    """

    # A is a (n x n)-matrix
    n = A[0, ].size

    # Copy A because it will be changed in-place
    A_copy = np.copy(A)
    # Permutation list
    P = []
    # Lower triangular matrix
    L = np.identity(n)
    # Diagonal matrix
    D = np.identity(n)

    if decompose_recursive(0, A_copy, P, L, D, A_copy):
        return P, L, D
    else:
        return None
