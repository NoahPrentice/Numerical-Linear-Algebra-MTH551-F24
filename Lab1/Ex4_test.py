from Ex4 import *

resolution = 200

def test_Aball(resolution: int):
    A = np.random.rand(2, 2)
    Aball(A, resolution)

test_Aball(resolution)