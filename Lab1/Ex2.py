import numpy as np
from Ex1 import inner_product


def residual_through_inner_product(
    orthonormal_vectors: list[np.ndarray], v: np.ndarray
) -> np.ndarray:
    """Computes the residual of a vector with respect to a set of orthonormal vectors by
    computing the inner product of each orthonormal vector with v.

    Parameters:
        orthonormal_vectors: a list of orthonormal "column vectors" (2D np.ndarray
            objects with 1 column).
        v: a "column vector" of the same size as the vectors in orthonormal_vectors.

    Returns:
        The residual of v with respect to orthonormal_vectors, that is, the result after
        applying Gram-Schmidt to v using the vectors in orthonormal_vectors.
    """
    for q in orthonormal_vectors:
        assert q.shape == v.shape

    residual = v
    for q in orthonormal_vectors:
        residual -= inner_product(q, v) * q
    return residual


def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes the outer product of two "column vectors" (2D np.ndarray with 1 column)
    
    Parameters:
        u: first "column vector"
        v: second "column vector"
    
    Returns:
        The outer product of u and v, uv^T.
    """
    matrix_list = []
    for row in u:
        row_i = []
        u_i = row[0]
        for column in v:
            v_j = column[0]
            row_i.append(u_i * v_j)
        matrix_list.append(row_i)
    return np.array(matrix_list)


def residual_through_outer_product(
    orthonormal_vectors: list[np.ndarray], v: np.ndarray
) -> np.ndarray:
    """Computes the residual of a vector with respect to a set of orthonormal vectors by
    computing the outer product of each orthonormal vector with itself.

    Parameters:
        orthonormal_vectors: a list of orthonormal "column vectors" (2D np.ndarray
            objects with 1 column).
        v: a "column vector" of the same size as the vectors in orthonormal_vectors.

    Returns:
        The residual of v with respect to orthonormal_vectors, that is, the result after
        applying Gram-Schmidt to v using the vectors in orthonormal_vectors.
    """
    for q in orthonormal_vectors:
        assert q.size == v.size

    residual = v
    for q in orthonormal_vectors:
        q_matrix = outer_product(q, q)
        residual -= q_matrix @ v
    return residual
