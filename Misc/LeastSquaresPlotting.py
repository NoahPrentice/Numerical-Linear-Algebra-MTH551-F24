import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2], [1]])
b = np.array([1, 1])


# Range of A is {(x, y) : 2y = x} or the line defined by y = 1/2 * x.
def f(x):
    return 0.5 * x


x, res, rnk, s = np.linalg.lstsq(A, b, rcond=None)
soln = A @ x
print(soln)
soln_coords = (soln[0], soln[1])
b_coords = (b[0], b[1])
residual = b - soln
residual_coords = (residual[0], residual[1])

line_x_values = [0.5, 2]
line_y_values = [f(a) for a in line_x_values]

plt.figure()
plt.plot(line_x_values, line_y_values, "-r", label="Range(A)")
plt.plot(soln_coords[0], soln_coords[1], "bo", label="Ax, least sq. sol'n")
plt.plot(b_coords[0], b_coords[1], "go", label="b")
plt.plot((soln_coords[0], b_coords[0]), (soln_coords[1], b_coords[1]), "-k")
plt.gca().set_aspect('equal')
plt.title("Least Squares Solution to Ax = b")
plt.legend()
plt.show()