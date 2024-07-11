import sys
import os
import numpy as np
from simplex import Simplex
from big_m import BigM


def main():
    c = np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)])
    a = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [-2, 1, -1, 0, -1, 1, 0],
        [0, 3, 1, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([4, 1, 9], dtype=np.float)
    simplex = Simplex(c, a, b, should_print_table=True)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)


if __name__ == '__main__':
    output_file_path = os.path.join('outp', 'main.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        sys.stdout = output_file
        main()
