import numpy as np
import solution as sol


def random_lower_int_matrix(n, max_val):
    """
    Generate random lower triangular matrix of size n.
    """
    a = np.random.random_integers(-max_val, max_val, size=(n, n))
    return np.identity(n) + np.tril(a, -1)


def random_upper_int_matrix(n, max_val):
    return random_lower_int_matrix(n, max_val).T


def random_diagonal_int_matrix(n, max_val):
    a = np.random.random_integers(1, max_val, size=n)
    return np.diag(a)


def random_block_diagonal_int_matrix(n, max_val):
    i = 0
    matrix_d = np.identity(n)
    while i < n - 1:
        block_size = 1 + (np.random.random() > 0.6)
        if block_size == 1:
            a = np.random.random_integers(-max_val, max_val)
            while a == 0:
                a = np.random.random_integers(-max_val, max_val)
            matrix_d[i, i] = a
        else:
            a, c, d = np.random.random_integers(-max_val, max_val, 3)
            while c == 0 or abs(c) <= 1.5 * max(abs(a), abs(d)):
                a, c, d = np.random.random_integers(-max_val, max_val, 3)
            matrix_d[i: i + 2, i: i + 2] = np.array([[a, c], [c, d]])
        i += block_size
    if i < n:
        a = np.random.random_integers(-max_val, max_val)
        while a == 0:
            a = np.random.random_integers(-max_val, max_val)
        matrix_d[i, i] = a
    return matrix_d


def random_vector(n, max_val):
    return np.random.random_integers(-max_val, max_val, size=(n, 1))


def test_forward_substitution():
    matrix_l = random_lower_int_matrix(10, 50)
    x = random_vector(10, 50)
    b = np.dot(matrix_l, x)
    return (sol.forward_substitution(matrix_l, b) == x).all()


def test_backward_substitution():
    matrix_u = random_upper_int_matrix(10, 50)
    x = random_vector(10, 50)
    b = np.dot(matrix_u, x)
    return (sol.backward_substitution(matrix_u, b) == x).all()


def test_diagonal_substitution():
    matrix_d = random_block_diagonal_int_matrix(10, 50)
    x = random_vector(10, 50)
    b = np.dot(matrix_d, x)
    return (sol.diagonal_substitution(matrix_d, b) == x).all()


def test():
    success = True
    for _ in range(1000):
        success = success and test_forward_substitution() and \
                  test_backward_substitution() and \
                  test_diagonal_substitution()
    print(success)


if __name__ == '__main__':
    test()


