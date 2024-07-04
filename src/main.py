from simplex import Simplex
import numpy as np
from big_m import BigM

if __name__ == '__main__':
    c = np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)])
    a = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [-2, 1, -1, 0, -1, 1, 0],
        [0, 3, 1, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([4, 1, 9], dtype=np.float)
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)
