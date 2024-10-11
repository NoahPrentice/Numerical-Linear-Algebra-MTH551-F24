from Ex2 import *
from time import perf_counter_ns
from Ex1_test import infinity_norm_difference
from scipy.linalg import orth
import statistics


def time_residual_thru_inner_product(
    orthonormal_vectors: list[np.ndarray], v: np.ndarray
) -> tuple[np.ndarray, float]:
    start = perf_counter_ns()
    residual = residual_through_inner_product(orthonormal_vectors, v)
    end = perf_counter_ns()
    execution_time = end - start
    return (residual, execution_time)


def time_residual_thru_outer_product(
    orthonormal_vectors: list[np.ndarray], v: np.ndarray
) -> tuple[np.ndarray, float]:
    start = perf_counter_ns()
    residual = residual_through_outer_product(orthonormal_vectors, v)
    end = perf_counter_ns()
    execution_time = end - start
    return (residual, execution_time)

def get_columns_from_matrix(matrix: np.ndarray) -> list[np.ndarray]:
    assert len(np.shape(matrix)) == 2

    columns = []
    number_of_columns = np.shape(matrix)[1]
    for i in range(number_of_columns):
        columns.append(matrix[:, i][None].T)
    return columns

def test_residual_methods(number_of_tests):
    differences_for_50_by_30 = []
    times_for_50_by_30 = []
    differences_for_50_by_50 = []
    times_for_50_by_50 = []
    for i in range(number_of_tests):
        A = np.random.rand(50, 30)
        orthonormal_vectors = get_columns_from_matrix(orth(A))
        v = np.random.rand(50, 1)
        inner_product_residual, inner_product_time = time_residual_thru_inner_product(orthonormal_vectors, v)
        outer_product_residual, outer_product_time = time_residual_thru_outer_product(orthonormal_vectors, v)
        differences_for_50_by_30.append(infinity_norm_difference(inner_product_residual, outer_product_residual))
        times_for_50_by_30.append(inner_product_time - outer_product_time)

        A = np.random.rand(50, 50)
        orthonormal_vectors = get_columns_from_matrix(orth(A))
        v = np.random.rand(50, 1)
        inner_product_residual, inner_product_time = time_residual_thru_inner_product(orthonormal_vectors, v)
        outer_product_residual, outer_product_time = time_residual_thru_outer_product(orthonormal_vectors, v)
        differences_for_50_by_50.append(infinity_norm_difference(inner_product_residual, outer_product_residual))
        times_for_50_by_50.append(inner_product_time - outer_product_time)
    
    print("The average infinity-norm difference for 50x30 matrices was " + str(statistics.fmean(differences_for_50_by_30)))
    print("On average, the inner product method was " + str(statistics.fmean(times_for_50_by_30)) + " ns slower than the outer product method for 50x30 matrices")
    print("The average infinity-norm difference for 50x50 matrices was " + str(statistics.fmean(differences_for_50_by_50)))
    print("On average, the inner product method was " + str(statistics.fmean(times_for_50_by_50)) + " ns slower than the outer product method for 50x50 matrices")

test_residual_methods(10)