import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

n = 50
number_of_matrices = 5


def error_with_no_fixes():
    U, X = np.linalg.qr(np.random.randn(n, n))
    V, X = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.flip(np.sort(np.random.rand(n, 1), None)))
    A = U @ S @ V.T

    U2, S2, V2h = np.linalg.svd(A)
    S2 = np.diag(S2)
    V2 = V2h.T

    print("U error: " + str(np.linalg.norm(U - U2, 2)))
    print("S error: " + str(np.linalg.norm(S - S2, 2)))
    print("V error: " + str(np.linalg.norm(V - V2, 2)))
    print("A error: " + str(np.linalg.norm(A - U2 @ S2 @ V2.T, 2)))


def error_with_fixed_svd():
    U, X = np.linalg.qr(np.random.randn(n, n))
    V, X = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.flip(np.sort(np.random.rand(n, 1), None)))
    A = U @ S @ V.T

    U2, S2, V2h = np.linalg.svd(A)
    V2 = V2h.T
    U, X = np.linalg.qr(np.random.randn(n, n))
    V, X = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.flip(np.sort(np.random.rand(n, 1), None)))

    A = U @ S @ V.T
    U2, S2, V2h = np.linalg.svd(A)
    S2 = np.diag(S2)
    V2 = V2h.T

    flip = np.sign(np.diag(U2.T @ U))
    U2 = U2 @ np.diag(flip)

    flip = np.sign(np.diag(V2.T @ V))
    V2 = V2 @ np.diag(flip)

    print("U error: " + str(np.linalg.norm(U - U2, 2)))
    print("S error: " + str(np.linalg.norm(S - S2, 2)))
    print("V error: " + str(np.linalg.norm(V - V2, 2)))
    print("A error: " + str(np.linalg.norm(A - U2 @ S2 @ V2.T, 2)))
    print("Cond(A): " + str(np.linalg.cond(A)))


def error_with_fixes_and_sixth_power():
    U, X = np.linalg.qr(np.random.randn(n, n))
    V, X = np.linalg.qr(np.random.randn(n, n))
    S = np.diag(np.flip(np.sort(np.random.rand(n, 1), None)))
    A = U @ S @ V.T

    U2, S2, V2h = np.linalg.svd(A)
    V2 = V2h.T
    U, X = np.linalg.qr(np.random.randn(n, n))
    V, X = np.linalg.qr(np.random.randn(n, n))
    S = np.power(np.diag(np.flip(np.sort(np.random.rand(n, 1), None))), 6)

    A = U @ S @ V.T
    U2, S2, V2h = np.linalg.svd(A)
    S2 = np.power(np.diag(S2), 6)
    V2 = V2h.T

    flip = np.sign(np.diag(U2.T @ U))
    U2 = U2 @ np.diag(flip)

    flip = np.sign(np.diag(V2.T @ V))
    V2 = V2 @ np.diag(flip)

    print("U error: " + str(np.linalg.norm(U - U2, 2)))
    print("S error: " + str(np.linalg.norm(S - S2, 2)))
    print("V error: " + str(np.linalg.norm(V - V2, 2)))
    print("A error: " + str(np.linalg.norm(A - U2 @ S2 @ V2.T, 2)))


def part_a():
    for i in range(number_of_matrices):
        error_with_no_fixes()


def part_b():
    for i in range(number_of_matrices):
        error_with_fixed_svd()


def part_c():
    for i in range(number_of_matrices):
        error_with_fixes_and_sixth_power()


part_b()
