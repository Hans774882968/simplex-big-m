from src.utils import get_unit_vector_to_idx, get_ith_unit_vector, is_infinite
from src.big_m import BigM


def test_get_unit_vector_to_idx():
    unit_vector_to_idx = get_unit_vector_to_idx(5)
    res = {
        (1, 0, 0, 0, 0): 0,
        (0, 1, 0, 0, 0): 1,
        (0, 0, 1, 0, 0): 2,
        (0, 0, 0, 1, 0): 3,
        (0, 0, 0, 0, 1): 4,
    }
    assert unit_vector_to_idx == res


def test_get_ith_unit_vector():
    second_unit_vec1 = get_ith_unit_vector(1, 5)
    res1 = [0, 1, 0, 0, 0]
    assert second_unit_vec1 == res1
    second_unit_vec2 = get_ith_unit_vector(0, 4)
    res2 = [1, 0, 0, 0]
    assert second_unit_vec2 == res2


def test_is_infinite():
    vals = [
        BigM(-1, 0), BigM(2, -2), BigM(0, 114514), BigM(0, float('inf')), float('-inf'), 1919810
    ]
    ans = [True, True, False, True, True, False]
    res = [is_infinite(v) for v in vals]
    assert res == ans
