import numpy as np
from src.simplex import Simplex
from src.big_m import BigM


def test_standard_input():
    c = np.array([2, 3, 0, 0, 0])
    a = np.array([
        [2, 2, 1, 0, 0],
        [4, 0, 0, 1, 0],
        [0, 5, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([12, 16, 15], dtype=np.float)
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    assert res_x == [3, 3, 0, 4, 0]
    assert res_val == 15


def test_directly_input_big_m_case1():
    c = np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)])
    a = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [-2, 1, -1, 0, -1, 1, 0],
        [0, 3, 1, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([4, 1, 9], dtype=np.float)
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    assert res_x == [0, 2.5, 1.5, 0, 0, 0, 0]
    assert res_val == 1.5


def test_directly_input_big_m_case2():
    c = np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)])
    a = np.array([
        [3, 0, 2, 1, 1, -1, 0],
        [-2, 1, -1, 0, -1, 1, 0],
        [6, 0, 4, 0, 3, -3, 1],
    ], dtype=np.float)
    b = np.array([3, 1, 6], dtype=np.float)
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    assert res_x == [0, 2.5, 1.5, 0, 0, 0, 0]
    assert res_val == 1.5


def test_directly_input_big_m_case3():
    c = np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)])
    a = np.array([
        [0, 0, 0, 1, -1 / 2, 1 / 2, -1 / 2],
        [0, 1, 1 / 3, 0, 0, 0, 1 / 3],
        [1, 0, 2 / 3, 0, 1 / 2, -1 / 2, 1 / 6],
    ], dtype=np.float)
    b = np.array([0, 3, 1], dtype=np.float)
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    assert res_x == [0, 2.5, 1.5, 0, 0, 0, 0]
    assert res_val == 1.5
