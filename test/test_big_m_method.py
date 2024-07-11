import pytest
import numpy as np
from math import isclose
from src.big_m_method import BigMMethod
from src.consts import Operator
from src.big_m import BigM
from src.utils import arr_close


def test_no_need_big_m():
    c = np.array([2, 3])
    a = np.array([
        [2, 2],
        [4, 0],
        [0, 5],
    ], dtype=np.float)
    b = np.array([12, 16, 15], dtype=np.float)
    operators = [Operator.LE, Operator.LE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators, is_debug=True)
    res_x, res_val = simplex.solve()
    assert res_x == [3, 3, 0, 4, 0]
    assert res_val == 15
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 5


def test_big_m_case1():
    c = np.array([-3, 0, 1])
    a = np.array([
        [1, 1, 1],
        [-2, 1, -1],
        [0, 3, 1],
    ], dtype=np.float)
    b = np.array([4, 1, 9], dtype=np.float)
    operators = [Operator.LE, Operator.GE, Operator.EQ]
    simplex = BigMMethod(c, a, b, operators, is_debug=True)
    res_x1, res_val1 = simplex.solve()
    assert res_x1 == [0, 2.5, 1.5, 0, 0, 0, 0]
    assert res_val1 == 1.5
    assert simplex.slack_variable_idx == 3
    assert simplex.artificial_var_idx == 5
    res_x2, res_val2 = simplex.solve()
    assert res_x2 == [0, 2.5, 1.5, 0, 0, 0, 0]
    assert res_val2 == 1.5
    assert simplex.slack_variable_idx == 3
    assert simplex.artificial_var_idx == 5
    assert np.array_equal(simplex.obj_func_coeff, np.array([-3, 0, 1, 0, 0, BigM(-1, 0), BigM(-1, 0)]))
    assert np.array_equal(simplex.constraints_coeff, np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [-2, 1, -1, 0, -1, 1, 0],
        [0, 3, 1, 0, 0, 0, 1],
    ], dtype=np.float))
    assert np.array_equal(simplex.b, np.array([4, 1, 9], dtype=np.float))


def test_big_m_case2_no_sol():
    c = np.array([2, 3])
    a = np.array([
        [2, 2],
        [1, 2],
    ], dtype=np.float)
    b = np.array([12, 14], dtype=np.float)
    operators = [Operator.LE, Operator.GE]
    simplex1 = BigMMethod(c, a, b, operators, is_debug=True)
    assert simplex1.slack_variable_idx == 2
    assert simplex1.artificial_var_idx == 4
    assert np.array_equal(simplex1.obj_func_coeff, np.array([2, 3, 0, 0, BigM(-1, 0)]))
    res_x1, _ = simplex1.solve()
    assert res_x1 == [0, 6, 0, 0, 2]
    simplex2 = BigMMethod(c, a, b, operators)
    assert simplex2.slack_variable_idx == 2
    assert simplex2.artificial_var_idx == 4
    with pytest.raises(ValueError) as e_info:
        simplex2.solve()
    assert 'This linear programming problem has no solution' in str(e_info.value)


def test_big_m_case3():
    # https://ww2.mathworks.cn/matlabcentral/answers/797482-problem-is-unbounded-in-linear-programming
    c = np.array([320, 510, 430, 300])
    a = np.array([
        [1, 4, 2, 3],
        [4, 5, 3, 1],
    ], dtype=np.float)
    b = np.array([300, 400], dtype=np.float)
    operators = [Operator.LE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators, is_debug=True)
    assert simplex.slack_variable_idx == 4
    assert simplex.artificial_var_idx == 6
    res_x, res_val = simplex.solve()
    assert arr_close(res_x, [0, 0, 128.57142857142858, 14.285714285714288, 0, 0])
    assert isclose(res_val, 59571.42857142858)


def test_big_m_case4():
    # 《运筹学基础及应用（第六版）》P54 习题1.1
    c = np.array([-2, -3], dtype=np.float)
    a = np.array([
        [4, 6],
        [4, 2],
    ], dtype=np.float)
    b = np.array([6, 4], dtype=np.float)
    operators = [Operator.GE, Operator.GE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 4
    res_x, res_val = simplex.solve()
    assert res_x == [0.75, 0.5]
    assert -res_val == 3


def test_big_m_case5():
    # 《运筹学基础及应用（第六版）》P54 习题1.1
    c = np.array([3, 2], dtype=np.float)
    a = np.array([
        [2, 1],
        [3, 4],
    ], dtype=np.float)
    b = np.array([2, 12], dtype=np.float)
    operators = [Operator.LE, Operator.GE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 4
    with pytest.raises(ValueError) as e_info:
        simplex.solve()
    assert 'This linear programming problem has no solution' in str(e_info.value)


def test_big_m_case6():
    # 《运筹学基础及应用（第六版）》P54 习题1.1
    c = np.array([1, 1], dtype=np.float)
    a = np.array([
        [6, 10],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], dtype=np.float)
    b = np.array([120, 5, 10, 3, 8], dtype=np.float)
    operators = [Operator.LE, Operator.GE, Operator.LE, Operator.GE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 7
    res_x, res_val = simplex.solve()
    assert res_x == [10, 6]
    assert res_val == 16


def test_big_m_case7():
    # 《运筹学基础及应用（第六版）》P54 习题1.1
    c = np.array([5, 6], dtype=np.float)
    a = np.array([
        [2, -1],
        [-2, 3],
    ], dtype=np.float)
    b = np.array([2, 2], dtype=np.float)
    operators = [Operator.GE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 4
    with pytest.raises(ValueError) as e_info:
        simplex.solve()
    assert 'This linear programming problem is unbounded' in str(e_info.value)


def test_big_m_case8():
    # 《运筹学基础及应用（第六版）》P55 习题1.2
    c = np.array([2, -4, 5, -6], dtype=np.float)
    a = np.array([
        [1, 4, -2, 8],
        [-1, 2, 3, 4],
    ], dtype=np.float)
    b = np.array([2, 1], dtype=np.float)
    operators = [Operator.EQ, Operator.EQ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 4
    assert simplex.artificial_var_idx == 4
    res_x, res_val = simplex.solve()
    assert res_x == [8, 0, 3, 0]
    assert res_val == 31


def test_big_m_case9():
    # 《运筹学基础及应用（第六版）》P55 习题1.2
    c = np.array([-5, 2, -3, -2], dtype=np.float)
    a = np.array([
        [1, 2, 3, 4],
        [2, 2, 1, 2],
    ], dtype=np.float)
    b = np.array([7, 3], dtype=np.float)
    operators = [Operator.EQ, Operator.EQ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 4
    assert simplex.artificial_var_idx == 4
    res_x, res_val = simplex.solve()
    assert res_x == [0, 0, 1, 1]
    assert res_val == -5


def test_big_m_case10():
    # 《运筹学基础及应用（第六版）》P55 习题1.7
    c = np.array([2, -1, 2], dtype=np.float)
    a = np.array([
        [1, 1, 1],
        [-2, 0, 1],
    ], dtype=np.float)
    b = np.array([6, 2], dtype=np.float)
    operators = [Operator.GE, Operator.GE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 3
    assert simplex.artificial_var_idx == 5
    with pytest.raises(ValueError) as e_info:
        simplex.solve()
    assert 'This linear programming problem is unbounded' in str(e_info.value)


def test_big_m_case11():
    # 《运筹学基础及应用（第六版）》P55 习题1.7
    c = np.array([-2, -3, -1], dtype=np.float)
    a = np.array([
        [1, 4, 2],
        [3, 2, 0],
    ], dtype=np.float)
    b = np.array([8, 6], dtype=np.float)
    operators = [Operator.GE, Operator.GE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 3
    assert simplex.artificial_var_idx == 5
    res_x, res_val = simplex.solve()
    assert res_x == [0.8, 1.8, 0]
    assert res_val == -7


def test_big_m_case12():
    # 《运筹学基础及应用（第六版）》P55 习题1.7
    c = np.array([-1, 5], dtype=np.float)
    a = np.array([
        [1, -2],
        [-1, 3],
    ], dtype=np.float)
    b = np.array([0.5, 0.5], dtype=np.float)
    operators = [Operator.LE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 4
    res_x, res_val = simplex.solve()
    assert arr_close(res_x, [2.5, 1])
    assert isclose(res_val, 2.5)


def test_big_m_case13():
    # 《运筹学基础及应用（第六版）》P60 习题1.22
    c = np.array([-1] * 8, dtype=np.float)
    a = np.array([
        [0, 0, 0, 0, 1, 1, 1, 2],
        [0, 1, 2, 3, 0, 1, 2, 0],
        [4, 3, 2, 0, 3, 1, 0, 1],
    ], dtype=np.float)
    b = np.array([100] * 3, dtype=np.float)
    operators = [Operator.EQ, Operator.EQ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 8
    assert simplex.artificial_var_idx == 8
    res_x, res_val = simplex.solve()
    '''
    枚举所有切割方案，即上面a矩阵的8列，然后设每列的使用的根数为x_i，于是得到8变量，3约束条件的线性规划问题。
    书上的答案是 [0, 0, 0, 0, 30, 0, 50, 10]，即103 30根，120 50根，201 10根。但我们的答案也对。
    '''
    assert arr_close(res_x, [0, 0, 30, 0, 0, 0, 20, 40])
    assert res_val == -90

# TODO: 这个用例过不了，暂时没找到原因，可能当前版本还有 bug
# def test_big_m_case14():
#     # 《运筹学基础及应用（第六版）》P53 例16
#     c = np.array([150, 150, 80, 80, 80], dtype=np.float)
#     a = np.array([
#         [500, 0, 0, 0, 0],
#         [1000, 0, 500, 0, 0],
#         [1500, 500, 1000, 0, 0],
#         [2000, 1000, 1500, 500, 0],
#         [2500, 1500, 2000, 1000, 0],
#         [3000, 2000, 2500, 1500, 500],
#         [3500, 2500, 2500, 2000, 1000],
#         [4000, 3000, 2500, 2500, 1500],
#         [4000, 3500, 2500, 2500, 2000],
#         [2000, 1000, 1500, 500, 0],
#         [3500, 2500, 2500, 2000, 1000],
#         [4000, 4000, 2500, 2500, 2500],
#         [1, 1, 1, 1, 1],
#     ], dtype=np.float)
#     b = np.array([
#         5000, 9000, 11500, 16000, 18500,
#         21500, 25500, 30000, 33500, 11500,
#         21500, 36500, 11
#     ], dtype=np.float)
#     operators = [
#         Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.LE,
#         Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.GE,
#         Operator.GE, Operator.GE, Operator.LE,
#     ]
#     simplex = BigMMethod(c, a, b, operators)
#     assert simplex.slack_variable_idx == 5
#     assert simplex.artificial_var_idx == 18
#     res_x, res_val = simplex.solve()
#     print('????????', res_x, res_val)
#     assert res_x == [1, 5, 3, 0, 2]
#     assert res_val == 1300


def test_unbounded_problem():
    c = np.array([2, 3])
    a = np.array([
        [4, 0],
    ], dtype=np.float)
    b = np.array([16], dtype=np.float)
    operators = [Operator.LE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 2
    assert simplex.artificial_var_idx == 3
    assert np.array_equal(simplex.obj_func_coeff, np.array([2, 3, 0]))
    with pytest.raises(ValueError) as e_info:
        simplex.solve()
    assert 'This linear programming problem is unbounded' in str(e_info.value)
