import numpy as np
from numpy import real, imag
import matplotlib.pyplot as plt

plt.figure()
for n in range(21, 31):
    roots = np.roots(np.poly(np.arange(1, n+1)))
    plt.plot(roots.real, roots.imag, "g*")
plt.title("Roots, z")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.show()