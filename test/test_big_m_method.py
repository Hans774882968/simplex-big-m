import pytest
import numpy as np
from math import isclose
from src.big_m_method import BigMMethod
from src.consts import Operator
from src.big_m import BigM, is_big_m_like
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
    # from https://ww2.mathworks.cn/matlabcentral/answers/797482-problem-is-unbounded-in-linear-programming
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


def test_big_m_case5_no_sol():
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


def test_big_m_case7_unbounded():
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


def test_big_m_case10_unbounded():
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
    operators = [Operator.EQ, Operator.EQ, Operator.EQ]
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


def test_big_m_case14():
    '''
    《运筹学基础及应用（第六版）》P53 例16 是整数规划问题，需要分支限界法（就是dfs剪枝+调用单纯形法）。
    感觉分支限界法不好实现，因此我手工调参运行了若干次，最终确定了额外增加的约束：
    x1 <= 1, x2 <= 6, y1 >= 3。于是得到了和书上一样的结果。
    '''
    c = np.array([150, 150, 80, 80, 80], dtype=np.float)
    a = np.array([
        [500, 0, 0, 0, 0],
        [1000, 0, 500, 0, 0],
        [1500, 500, 1000, 0, 0],
        [2000, 1000, 1500, 500, 0],
        [2500, 1500, 2000, 1000, 0],
        [3000, 2000, 2500, 1500, 500],
        [3500, 2500, 2500, 2000, 1000],
        [4000, 3000, 2500, 2500, 1500],
        [4000, 3500, 2500, 2500, 2000],
        [2000, 1000, 1500, 500, 0],
        [3500, 2500, 2500, 2000, 1000],
        [4000, 4000, 2500, 2500, 2500],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float)
    b = np.array([
        5000, 9000, 11500, 16000, 18500,
        21500, 25500, 30000, 33500, 11500,
        21500, 36500, 11, 1, 6,
        3,
    ], dtype=np.float)
    operators = [
        Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.LE,
        Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.GE,
        Operator.GE, Operator.GE, Operator.LE, Operator.LE, Operator.LE,
        Operator.GE,
    ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 5
    assert simplex.artificial_var_idx == 21
    res_x, res_val = simplex.solve()
    assert arr_close(res_x, [1, 5, 3, 0, 2])
    assert is_big_m_like(res_val) and abs(res_val.a) < 1e-14 and res_val.b == 1300


def test_big_m_case15():
    '''
    《运筹学基础及应用（第六版）》P58 习题1.14
    设第i个月租j个月的合同的租借面积为 xij ，则有 i+j<=5 ，得共有10个变量。
    目标函数：min(2800*sum(xi1, i=1~4) + 4500*sum(xi2, i=1~3) + 6000*sum(xi3, i=1~2) + 7300*x14)
    约束条件：能覆盖到第i个月的合同的总面积大于等于所需仓库面积。
    x11+x12+x13+x14 >= 15
    x12+x13+x14+x21+x22+x23 >= 10
    x13+x14+x22+x23+x31+x32 >= 20
    x14+x23+x32+x41 >= 12
    看下面代码的 a 矩阵，比较优美，分别由4*4~1*1的上三角阵构成。
    解得x11 = 5, x14 = 10, x31 = 8, x32 = 2，这表示第1个月租1个月合同500平，租4个月合同1000平，
    第3个月租1个月合同800平，租2个月合同200平。于是第1~4个月分别覆盖到500 + 1000，1000，1000 + 800 + 200，
    1000 + 200平，恰好都等于每个月所需仓库面积。
    '''
    c = np.array([-2800, -4500, -6000, -7300, -2800, -4500, -6000, -2800, -4500, -2800], dtype=np.float)
    a = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
    ], dtype=np.float)
    b = np.array([15, 10, 20, 12], dtype=np.float)
    operators = [Operator.GE, Operator.GE, Operator.GE, Operator.GE]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 10
    assert simplex.artificial_var_idx == 14
    res_x, res_val = simplex.solve()
    assert res_x == [5, 0, 0, 10, 0, 0, 0, 8, 2, 0]
    assert res_val == -118400


def test_big_m_case16_method1():
    '''
    《运筹学基础及应用（第六版）》P58 习题1.16
    参考：https://blog.csdn.net/weixin_45755831/article/details/113216833
    pc = process costing
    注意书上的答案思路是我下面提供的方法二，但给到的答案是方法一的，只能说书作者抄别人的书都抄不明白…
    方法一：设产品1经过A工序的数量为x1~x3，经过B工序的数量为x4~x5，则有x1+x2+x3=x4+x5。
    同理设x6~x10，有x6+x7=x8，x9=x10。目标函数：获利=(售价-原料费)*产品总数-设备加工费。所以
    max((x1 + x2) + 1.65 * x8 + 2.3 * x9 - pc[0] * 5 * x1 - pc[1] * 7 * x2 - pc[2] * 6 * x3
    - pc[3] * 4 * x4 - pc[4] * 7 * x5 - pc[0] * 10 * x6 - pc[1] * 9 * x7 - pc[2] * 8 * x8
    - pc[1] * 12 * x9 - pc[3] * 11 * x10)
    约束条件：（1）不超过设备有效台时。（2）上面3个等式。
    5 * x1 + 10 * x6 <= 6000
    7 * x2 + 9 * x7 + 12 * x9 <= 10000
    6 * x3 + 8 * x8 <= 4000
    4 * x4 + 11 * x10 <= 7000
    7 * x5 <= 4000
    在下面的代码里，我把x10都换成x9，减少一个变量。
    最初的解：
    [1200.0, 230.0492610837435, 0, 858.620689655172, 571.4285714285714, 0, 500.0, 500.0, 324.1379310344829] 1146.5665024630541
    但原问题是一个整数规划问题，于是我多尝试运行了几次，最后手工加了3条限制条件，成功缚住苍龙。

    方法二：生产产品1共2*3种方案，设经A1、B1加工的产品1数量为x11，同理A1、B2到A2、B3的产品数量为x12~x16。
    同理设x21, x22, x3。设方法一的变量以y为开头，则有x11+x12+x13=y1，x14+x15+x16=y2，x11+x14=y3等。
    目标函数和方法一同理推导，在此就不完整写出来了。仅以设备A2的加工费为例：
    pc[1] * 7 * x14 + pc[1] * 9 * x22 + pc[1] * 12 * x3
    约束条件也和方法一同理推导，仅以设备B1为例：
    6 * (x11 + x14) + 8 * (x21 + x22) <= 4000
    最初的解：
    [0, 858.620689655172, 341.37931034482796, 0, 0, 230.0492610837435, 0, 500.0, 324.1379310344829] 1146.5665024630541
    后面我同样加了3条限制条件，得最终解。
    '''
    pc = [0.05, 0.0321, 0.0625, 783 / 7000, 0.05]
    c = np.array([
        1 - pc[0] * 5, 1 - pc[1] * 7, -pc[2] * 6, -pc[3] * 4, -pc[4] * 7,
        -pc[0] * 10, -pc[1] * 9, 1.65 - pc[2] * 8, 2.3 - pc[1] * 12 - pc[3] * 11
    ], dtype=np.float)
    a = np.array([
        [5, 0, 0, 0, 0, 10, 0, 0, 0],
        [0, 7, 0, 0, 0, 0, 9, 0, 12],
        [0, 0, 6, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 11],
        [0, 0, 0, 0, 7, 0, 0, 0, 0],
        [1, 1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, -1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([6000, 10000, 4000, 7000, 4000, 0, 0, 230, 859, 324], dtype=np.float)
    operators = [
        Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.LE,
        Operator.EQ, Operator.EQ, Operator.EQ, Operator.EQ, Operator.EQ,
    ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 9
    assert simplex.artificial_var_idx == 14
    res_x, res_val = simplex.solve()
    assert res_x == [1200, 230, 0, 859, 571, 0, 500, 500, 324]
    assert res_val == 1146.4142


def test_big_m_case16_method2():
    pc = [0.05, 0.0321, 0.0625, 783 / 7000, 0.05]
    c = np.array([
        1 - pc[0] * 5 - pc[2] * 6, 1 - pc[0] * 5 - pc[3] * 4, 1 - pc[0] * 5 - pc[4] * 7,
        1 - pc[1] * 7 - pc[2] * 6, 1 - pc[1] * 7 - pc[3] * 4, 1 - pc[1] * 7 - pc[4] * 7,
        1.65 - pc[0] * 10 - pc[2] * 8, 1.65 - pc[1] * 9 - pc[2] * 8, 2.3 - pc[1] * 12 - pc[3] * 11
    ], dtype=np.float)
    a = np.array([
        [5, 5, 5, 0, 0, 0, 10, 0, 0],
        [0, 0, 0, 7, 7, 7, 0, 9, 12],
        [6, 0, 0, 6, 0, 0, 8, 8, 0],
        [0, 4, 0, 0, 4, 0, 0, 0, 11],
        [0, 0, 7, 0, 0, 7, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([6000, 10000, 4000, 7000, 4000, 859, 230, 324], dtype=np.float)
    operators = [
        Operator.LE, Operator.LE, Operator.LE, Operator.LE, Operator.LE,
        Operator.EQ, Operator.EQ, Operator.EQ,
    ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 9
    assert simplex.artificial_var_idx == 14
    res_x, res_val = simplex.solve()
    assert res_x == [0, 859, 341, 0, 0, 230, 0, 500, 324]
    assert res_val == 1146.4142
    assert sum(res_x[0:3]) == 1200
    assert sum(res_x[3:6]) == 230
    assert res_x[0] + res_x[3] == 0
    assert res_x[1] + res_x[4] == 859
    assert res_x[2] + res_x[5] == 571


def test_big_m_case17():
    '''
    《运筹学基础及应用（第六版）》P59 习题1.21
    设全日制员工12到13点用餐、13到14点用餐的数量分别为x1, x2，各个批次的非全日制员工数量分别为y1~y6。
    约束的系数矩阵按列填写即可做到快速填写。
    书上答案给的res_x = [3, 3, 0, 0, 3, 0, 0, 2], res_val = -1840是错的，答案应该是
    res_x = [0, 0, 4, 1, 5, 0, 0, 8], res_val = -1440
    '''
    c = np.array([-240] * 2 + [-80] * 6, dtype=np.float)
    a = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([4, 5, 6, 6, 5, 6, 8, 8], dtype=np.float)
    operators = [
        Operator.GE, Operator.GE, Operator.GE, Operator.GE, Operator.GE,
        Operator.GE, Operator.GE, Operator.GE,
    ]
    simplex = BigMMethod(c, a, b, operators)
    assert simplex.slack_variable_idx == 8
    assert simplex.artificial_var_idx == 16
    res_x, res_val = simplex.solve()
    assert res_x == [0, 0, 4, 1, 5, 0, 0, 8]
    assert res_val == -1440


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
