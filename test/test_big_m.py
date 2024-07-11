from src.big_m import BigM


def test_unary():
    a1 = BigM(-114, -514)
    assert -a1 == BigM(114, 514)
    a2 = BigM(0, float('inf'))
    assert -a2 == BigM(0, float('-inf'))


def test_big_m_cmp():
    a = BigM(1, 2)
    b = BigM(3, 1)
    c = BigM(1, 2)
    assert a < b
    assert a <= b
    assert a <= c
    assert a == c
    assert b > a
    assert b >= a
    assert c >= a
    assert not (a != c)
    assert a != b


def test_big_m_and_number_cmp():
    a1 = BigM(1, -10)
    v = 100
    assert a1 > v
    assert a1 >= v
    assert a1 != v
    a2 = BigM(0, 20)
    assert a2 < v
    assert a2 <= v
    a3 = BigM(0, 100)
    assert a3 == v
    assert a3 <= v
    assert a3 >= v
    assert v <= a3
    assert v >= a3
    assert not (a3 != v)
    a4 = BigM(0, 101)
    assert a4 > v
    assert a4 >= v
    assert v < a4
    assert v <= a4
    a5 = BigM(-1, 1000)
    assert a5 < v
    assert a5 <= v
    assert a5 != v
    assert v >= a5
    assert v > a5


def test_add():
    a = BigM(1, 2)
    b = BigM(3, 1)
    assert a + b == BigM(4, 3)
    assert b + a == BigM(4, 3)
    k = 18
    assert a + k == BigM(1, 20)
    assert k + a == BigM(1, 20)


def test_sub():
    a = BigM(1, 2)
    b = BigM(3, 1)
    assert a - b == BigM(-2, 1)
    assert b - a == BigM(2, -1)
    k = 22
    assert a - k == BigM(1, -20)
    assert k - a == BigM(-1, 20)


def test_mul():
    a = BigM(1, 2)
    b = BigM(3, 1)
    assert a * b == BigM(10, 2)
    assert b * a == BigM(10, 2)
    k = 22
    assert b * k == BigM(66, 22)
    assert k * b == BigM(66, 22)


def test_str():
    a1 = BigM(1, 2) - BigM(1, 3)
    assert str(a1) == '-1'
    a2 = BigM(-3, -1)
    assert str(a2) == '-3 * M + -1'
    a3 = BigM(108, 0)
    assert str(a3) == '108 * M'


def test_min_max_sum():
    arr = [BigM(1, 20), BigM(0, 100), 101, BigM(-1, 1000)]
    assert min(arr) == BigM(-1, 1000)
    assert max(arr) == BigM(1, 20)
    sm = sum(arr)
    assert sm == BigM(0, 1221) and sm == 1221
