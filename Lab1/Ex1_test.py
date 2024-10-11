from Ex1 import *
from time import perf_counter_ns
import statistics


def time_sum_of_columns(A: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter_ns()
    result = multiplication_thru_sum_of_columns(A, x)
    end = perf_counter_ns()
    execution_time = end - start
    return (result, execution_time)


def time_inner_product(A: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    start = perf_counter_ns()
    result = multiplication_thru_inner_prodcut(A, x)
    end = perf_counter_ns()
    execution_time = end - start
    return (result, execution_time)


def infinity_norm_difference(u: np.ndarray, v: np.ndarray) -> float:
    return np.linalg.norm(u - v, np.inf)


def test_matrix_multiplication_methods(number_of_tests: int):

    two_norm_differences = []
    two_time_differences = []

    hundred_norm_differences = []
    hundred_time_differences = []

    for i in range(number_of_tests):
        # m = 2
        A = np.random.rand(2, 2)
        x = np.random.rand(2, 1)
        sum_of_columns_vector, sum_of_columns_time = time_sum_of_columns(A, x)
        inner_product_vector, inner_product_time = time_inner_product(A, x)
        two_norm_differences.append(
            infinity_norm_difference(sum_of_columns_vector, inner_product_vector)
        )
        two_time_differences.append(sum_of_columns_time - inner_product_time)

        # m = 100
        A = np.random.rand(100, 100)
        x = np.random.rand(100, 1)
        sum_of_columns_vector, sum_of_columns_time = time_sum_of_columns(A, x)
        inner_product_vector, inner_product_time = time_inner_product(A, x)
        hundred_norm_differences.append(
            infinity_norm_difference(sum_of_columns_vector, inner_product_vector)
        )
        hundred_time_differences.append(sum_of_columns_time - inner_product_time)

    print(
        "The average infinity-norm difference between the results of the two methods for 2x2 matrices is "
        + str(statistics.fmean(two_norm_differences))
    )
    print(
        "On average, 2x2 matrix multiplication through a weighted sum of columns was "
        + str(statistics.fmean(two_time_differences))
        + " ns slower than through an inner product"
    )

    print("")

    print(
        "The average infinity-norm difference between the results of the two methods for 100x100 matrices is "
        + str(statistics.fmean(hundred_norm_differences))
    )
    print(
        "On average, 100x100 matrix multiplication through a weighted sum of columns was "
        + str(statistics.fmean(hundred_time_differences))
        + " ns slower than through an inner product"
    )


test_matrix_multiplication_methods(10)
