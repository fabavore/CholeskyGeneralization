import decomposition as dc
import test_decomposition
import solution
import test_solution
import random_matrices as rm


if __name__ == '__main__':
    print(rm.symmetric(5, 50), "\n")
    print(rm.symmetric(5, 50, True), "\n")
    print(rm.lower_triangular(5, 50), "\n")
    print(rm.lower_triangular(5, 50, True), "\n")
    print(rm.upper_triangular(5, 50), "\n")
    print(rm.upper_triangular(5, 50, True), "\n")
    print(rm.block_diagonal(5, 50, 0.5, dc.eps, dc.epsilon), "\n")
    print(rm.block_diagonal(5, 50, 0.5, dc.eps, dc.epsilon, True), "\n")
