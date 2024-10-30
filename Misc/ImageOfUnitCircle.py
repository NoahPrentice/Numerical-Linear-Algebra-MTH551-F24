import math
import matplotlib.pyplot as plt
import numpy as np

A_list = [[0, 2], [0, 0]]
A = np.array(A_list)
B_list = [[0, 1], [-1, 0]]
B = np.array(B_list)
C_list = [[2, 1], [1, 3]]
C = np.array(C_list)

t = np.arange(0, 2 * np.pi, 0.01)
x = np.cos(t)
y = np.sin(t)

x_A = []
y_A = []
x_B = []
y_B = []
x_C = []
y_C = []
for i in range(x.size):
    vector = np.array([[x[i]], [y[i]]])
    A_vector = np.matmul(A, vector)
    x_A.append(A_vector[0][0])
    y_A.append(A_vector[1][0])
    B_vector = np.matmul(B, vector)
    x_B.append(B_vector[0][0])
    y_B.append(B_vector[1][0])
    C_vector = np.matmul(C, vector)
    x_C.append(C_vector[0][0])
    y_C.append(C_vector[1][0])

plt.figure(1)

plot1 = plt.subplot(221, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
plot1.scatter(x, y, marker=".")
plot1.set_title("The unit circle, S")

plot2 = plt.subplot(222, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
plot2.scatter(x_A, y_A, marker=".")
plot2.set_title("AS for (a)")

plot3 = plt.subplot(223, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
plot3.scatter(x_B, y_B, marker=".")
plot3.set_title("AS for (b)")

plot4 = plt.subplot(224, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
plot4.scatter(x_C, y_C, marker=".")
plot4.set_title("AS for (c)")

plt.tight_layout()
plt.show()
