import numpy as np
import decomposition as dc


def test_decompose(matrix_a):
    try:
        permutations, matrix_l, matrix_d = dc.decompose(matrix_a)
    except TypeError:
        print("Singular matrix:\n", matrix_a)
        return -1

    dc.permute_symmetric(matrix_a, permutations)

    probe = np.dot(np.dot(matrix_l, matrix_d), matrix_l.T)

    if np.array_equal(matrix_a, probe):
        return 0
    elif np.allclose(matrix_a, probe):
        return 1
    else:
        return 2





def test():
    c_1, c_2, c_3, c_4 = 0, 0, 0, 0
    for _ in range(1000):
        t = test_decompose(random_symmetric_int_matrix(10, 50))
        if t == 0:
            c_1 += 1
        elif t == 1:
            c_2 += 1
        elif t == 2:
            c_3 += 1
        else:
            c_4 += 1
    print("Success: ", c_1,
          ", Close enough: ", c_2,
          ", Not even close: ", c_3,
          ", Singular matrix: ", c_4)


if __name__ == '__main__':
    test()
