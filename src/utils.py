from typing import Union, List
from big_m import BigM, is_big_m_like
from math import isinf, isclose


def get_unit_vector_to_idx(m: int):
    return dict([[tuple([0] * i + [1] + [0] * (m - i - 1)), i] for i in range(m)])


def get_ith_unit_vector(pos: int, m: int, k: Union[int, float] = 1):
    return [0] * pos + [k] + [0] * (m - pos - 1)


def is_infinite(v: Union[int, float, BigM]):
    if is_big_m_like(v):
        if v.a != 0:
            return True
        return isinf(v.b)
    return isinf(v)


def arr_close(a: List[Union[int, float]], expected: List[Union[int, float]]):
    return all([isclose(v1, v2) for v1, v2 in zip(a, expected)])
