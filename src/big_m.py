class BigM:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            if self.a < 0:
                return True
            if self.a == 0:
                return self.b < other
            return False
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        if self.a < other.a:
            return True
        elif self.a == other.a:
            return self.b < other.b
        return False

    def __eq__(self, other: object) -> bool:
        if is_big_m_like(other):
            return self.a == other.a and self.b == other.b
        if isinstance(other, (int, float)):
            return self.a == 0 and self.b == other
        return False

    def __le__(self, other: object) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: object) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: object) -> bool:
        return not self.__lt__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __neg__(self):
        return BigM(-self.a, -self.b)

    def __add__(self, other: object):
        if isinstance(other, (int, float)):
            return BigM(self.a, self.b + other)
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        return BigM(self.a + other.a, self.b + other.b)

    def __radd__(self, other: object):
        return self.__add__(other)

    def __sub__(self, other: object):
        if isinstance(other, (int, float)):
            return BigM(self.a, self.b - other)
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        return BigM(self.a - other.a, self.b - other.b)

    def __rsub__(self, other: object):
        if isinstance(other, (int, float)):
            return BigM(-self.a, other - self.b)
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        return BigM(other.a - self.a, other.b - self.b)

    def __mul__(self, other: object):
        if isinstance(other, (int, float)):
            return BigM(self.a * other, self.b * other)
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        return BigM(self.a * other.a + self.b * other.a + self.a * other.b, self.b * other.b)

    def __rmul__(self, other: object):
        return self.__mul__(other)

    def __str__(self) -> str:
        if self.a == 0:
            return str(self.b)
        if self.b == 0:
            return f'{self.a} * M'
        return f'{self.a} * M + {self.b}'


# Note: don't use isinstance(v, BigM) in test_utils.py etc. because src.big_m and big_m are distinct, which leads to logic errors
def is_big_m_like(v: object):
    return hasattr(v, 'a') and hasattr(v, 'b')


if __name__ == '__main__':
    a = BigM(1, 2)
    b = BigM(3, 1)
    c = BigM(1, 2)
    print(a <= b)
    print(a == c)
    print(a + b)
    print(a - b)
    print(a - c)
