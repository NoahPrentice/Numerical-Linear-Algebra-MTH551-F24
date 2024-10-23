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
reciprocal_matrix = np.array(hilb_matrix, copy=True)
for i in range(reciprocal_matrix.shape[0]):
    for j in range(reciprocal_matrix.shape[1]):
        reciprocal_matrix[i, j] = 1 / reciprocal_matrix[i, j]

matrices = {
    "rand_matrix": rand_matrix,
    "randn_matrix": randn_matrix,
    "hilb_matrix": hilb_matrix,
    "invhilb_matrix": invhilb_matrix,
    "reciprocal_matrix": reciprocal_matrix,
}
first_matrices = {
    "rand_matrix": rand_matrix,
    "randn_matrix": randn_matrix,
    "hilb_matrix": hilb_matrix,
}

second_matrices = {
    "invhilb_matrix": invhilb_matrix,
    "reciprocal_matrix": reciprocal_matrix,
}


def write_singular_values_to_file():
    singular_values_file = open("Ex1SingularValues.txt", "w")
    singular_values = []
    header = ""
    for matrix_name in first_matrices:
        matrix = matrices[matrix_name]
        header += matrix_name + " & "
        singular_values.append(linalg.svdvals(matrix))
    singular_values_file.write(header[:-2] + "\\\\ \n")
    singular_values = np.array(singular_values).T
    for row in singular_values:
        next_line = ""
        for value in row:
            next_line += str(value) + " & "
        singular_values_file.write(next_line[:-2] + "\\\\ \n")

    singular_values_file.write("\n \n")
    singular_values = []
    header = ""
    for matrix_name in second_matrices:
        matrix = matrices[matrix_name]
        header += matrix_name + " & "
        singular_values.append(linalg.svdvals(matrix))
    singular_values_file.write(header[:-2] + "\\\\ \n")
    singular_values = np.array(singular_values).T
    for row in singular_values:
        next_line = ""
        for value in row:
            next_line += str(value) + " & "
        singular_values_file.write(next_line[:-2] + "\\\\ \n")

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
        + str("{:0.2e}".format(np.linalg.det(matrix)))
        + ", cond = "
        + str("{:0.2e}".format(np.linalg.cond(matrix)))
    )
    plt.ylabel(r"$\sigma_i$")
    plt.xlabel(r"$i$")
    plt.show()
