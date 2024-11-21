import math
import numpy as np

epsilon = math.pow(10, -10)

a = np.array([[1, 1], [epsilon, 0], [0, epsilon]])
a_star = np.transpose(a)
b = np.array([[-epsilon], [1 + epsilon], [1 - epsilon]])


def part_b():
    return np.linalg.lstsq(a, b)


def part_c():
    return np.linalg.lstsq(a_star @ a, a_star @ b)


def part_d():
    # numpy produces LL^* Cholesky factorization instead of R^*R.
    l, l_star = np.linalg.cholesky(a_star @ a)
    r = np.transpose(l_star)
    r_star = np.transpose(l)

    w = np.linalg.solve(r_star, a_star @ b)
    return np.linalg.solve(r, w)


def part_e():
    q, r = np.linalg.qr(a)
    q_star = np.transpose(q)
    return np.linalg.solve(r, q_star @ b)


def part_f():
    u, s, v_star = np.linalg.svd(a, full_matrices=False)
    u_star = np.transpose(u)
    s = np.diag(s)  # singular values are put in a 1d array by default
    v = np.transpose(v_star)

    w = np.linalg.solve(s, u_star @ b)
    return v @ w
