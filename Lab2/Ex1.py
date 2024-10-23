import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg

m = 20
np.random.seed(0)
rand_matrix = np.random.rand(m, m)
randn_matrix = np.random.randn(m, m)
hilb_matrix = linalg.hilbert(m)
invhilb_matrix = linalg.invhilbert(m)
reciprical_matrix = np.array(invhilb_matrix, copy=True)
for i in range(reciprical_matrix.shape[0]):
    for j in range(reciprical_matrix.shape[1]):
        reciprical_matrix[i, j] = 1 / reciprical_matrix[i, j]

matrices = {
    "rand_matrix": rand_matrix,
    "randn_matrix": randn_matrix,
    "hilb_matrix": hilb_matrix,
    "invhilb_matrix": invhilb_matrix,
    "reciprical_matrix": reciprical_matrix,
}


def write_singular_values_to_file():
    singular_values_file = open("Ex1SingularValues.txt", "w")
    singular_values_print_list = []
    for matrix_name in matrices:
        singular_values_print_list.append(matrix_name + ":\n")
        matrix = matrices[matrix_name]
        for singular_value in linalg.svdvals(matrix):
            singular_values_print_list.append(str(singular_value) + "\n")
    singular_values_file.writelines(singular_values_print_list)


def plot_singular_values_of_matrix(matrix_name):
    matrix = matrices[matrix_name]
    singular_values = linalg.svdvals(matrix)
    plt.figure()
    plt.scatter(range(singular_values.size), singular_values)
    plt.title(
        matrix_name
        + ", norm = "
        + str("{:0.2e}".format(np.linalg.norm(matrix, 2)))
        + ", det = "
        + str((np.linalg.det(matrix)))
        + ", cond = "
        + str("{:0.2e}".format(np.linalg.cond(matrix)))
    )
    plt.ylabel(r"$\sigma_i$")
    plt.xlabel(r"$i$")
    plt.show()
