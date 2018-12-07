import numpy as np


def matrix(n, max_val, int_val=False):
    shape = (n, n)
    return np.random.random_integers(-max_val, max_val, shape) if int_val \
        else 2 * (np.random.random_sample(shape) - 0.5 * np.ones(shape)) * max_val


def single_entry(max_val, eps=0, int_val=False):
    while True:
        a = np.random.random_integers(-max_val, max_val) if int_val else 2 * (np.random.random() - 0.5) * max_val
        if abs(a) > eps:
            return a


def symmetric(n, max_val, int_val=False):
    a = matrix(n, max_val, int_val)
    return np.tril(a) + np.tril(a, -1).T


def lower_triangular(n, max_val, int_val=False):
    a = matrix(n, max_val, int_val)
    return np.identity(n) + np.tril(a, -1)


def upper_triangular(n, max_val, int_val=False):
    return lower_triangular(n, max_val, int_val).T


def block_diagonal(n, max_val, ratio, eps, epsilon, int_val=False):
    i = 0
    matrix_d = np.identity(n)
    while i < n - 1:
        block_size = 1 + (np.random.random() < ratio)
        if block_size == 1:
            matrix_d[i, i] = single_entry(max_val, eps, int_val)
        else:  # block_size == 2
            c = single_entry(max_val, 3 * eps, int_val)
            current_max_val = c / (1 + epsilon)
            a, d = single_entry(current_max_val, eps, int_val), single_entry(current_max_val, eps, int_val)
            matrix_d[i:i + 2, i:i + 2] = np.array([[a, c], [c, d]])
        i += block_size
    if i < n:
        matrix_d[i, i] = single_entry(max_val, eps, int_val)
    return matrix_d
