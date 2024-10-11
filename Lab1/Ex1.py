import numpy as np
import math


def inner_product(u: np.ndarray, v: np.ndarray) -> float:
    """Calculates the inner product of two vectors of equal size.

    Parameters:
        u: first vector, as a "column vector" (i.e., a 2D np.ndarray with 1 column).
        v: second vector, also as a "column vector."

    Returns:
        The inner product of u and v.
    """
    assert u.size == v.size

    inner_product = 0
    for i in range(u.size):
        inner_product += u[i][0] * v[i][0]
    return inner_product


def multiplication_thru_inner_prodcut(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Left-multiplies a vector by a matrix by computing inner products of the rows of
    the matrix with the vector.

    Parameters:
        A: matrix (2D np.ndarray) with, say, n columns.
        x: "column vector" (2D np.ndarray with 1 column) of length n.

    Returns:
        The product Ax, as a "column vector."
    """
    assert len(A.shape) == 2
    assert A.shape[1] == x.size

    Ax = []
    for i in range(x.size):
        row_i_as_array = A[i, :][None]
        Ax.append([inner_product(row_i_as_array.T, x)])
    return np.array(Ax)


def multiplication_thru_sum_of_columns(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Left-multiplies a vector by a matrix by computing a weighted sum of the columns of
    the matrix.

    Parameters:
        A: matrix (2D np.ndarray) with, say, n columns.
        x: "column vector" (2D np.ndarray with 1 column) of length n.

    Returns:
        The product Ax, as a "column vector."
    """
    assert len(A.shape) == 2
    assert A.shape[1] == x.size

    A_columns = [A[:, [i]] for i in range(A.shape[1])]
    weighted_sum_of_columns = A_columns[0] * x[0][0]
    for i in range(1, x.size):
        weighted_sum_of_columns += A_columns[i] * x[i][0]
    return weighted_sum_of_columns
