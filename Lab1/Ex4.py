import math
import numpy as np
import matplotlib.pyplot as plt


def Aball(A: np.ndarray, M: int) -> None:
    """Plots the image of the unit ball under matrix A with resolution M."""
    t = [i / M for i in range(M)] + [0.0]
    x = [math.cos(2 * math.pi * t_i) for t_i in t]
    y = [math.sin(2 * math.pi * t_i) for t_i in t] 
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for i in range(M + 1):
        old_x = x[i]
        old_y = y[i]
        vector = np.array([[old_x], [old_y]])
        vector = np.matmul(A, vector)
        new_x = vector[0][0]
        new_y = vector[1][0]
        x[i] = new_x
        y[i] = new_y
        x_min = min(x_min, new_x)
        y_min = min(y_min, new_y)
        x_max = max(x_max, new_x)
        y_max = max(y_max, new_y)

    plt.figure(1)
    myplot = plt.subplot(111, xlim=(x_min-0.2, x_max+0.2), ylim=(y_min-0.2, y_max+0.2), aspect="equal")
    myplot.plot(x, y, "-")
    print_A = np.round(A, 2)
    det_A = round(np.linalg.det(A), 2)
    eig_A = [round(eig, 2) for eig in np.linalg.eig(A).eigenvalues]
    cond_A = round(np.linalg.cond(A), 2)
    display_text = (
        " A: "
        + str(print_A)
        + "\n det(A): "
        + str(det_A)
        + "\n eig(A): "
        + str(eig_A)
        + "\n cond(A): "
        + str(cond_A)
    )
    myplot.text(
        1.01,
        0.7,
        display_text,
        transform=myplot.transAxes,
        bbox=dict(facecolor="grey", alpha=0.2),
    )
    plt.show()
