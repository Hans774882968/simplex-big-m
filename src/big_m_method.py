from typing import List
import sys
import os
import numpy as np
from numpy import ndarray
from big_m import BigM, is_big_m_like
from simplex import Simplex
from utils import get_unit_vector_to_idx, get_ith_unit_vector
from consts import Operator


class BigMMethod(Simplex):
    def __init__(
            self,
            obj_func_coeff: ndarray,
            constraints_coeff: ndarray,
            b: ndarray,
            operators: List[str],
            should_print_table: bool = False,
            is_debug: bool = False) -> None:
        self.slack_variable_idx = len(obj_func_coeff)
        obj_func_coeff, constraints_coeff = self._to_standard_model(obj_func_coeff, constraints_coeff, operators)
        self.artificial_var_idx = len(obj_func_coeff)
        obj_func_coeff, constraints_coeff = self._add_artificial_variables(obj_func_coeff, constraints_coeff)
        super().__init__(obj_func_coeff, constraints_coeff, b, should_print_table, is_debug)

    def _to_standard_model(
            self,
            obj_func_coeff: ndarray,
            constraints_coeff: ndarray,
            operators: List[str]):
        m = len(constraints_coeff)
        for i, op in enumerate(operators):
            if op == Operator.LE:
                unit_vec = np.reshape(get_ith_unit_vector(i, m), (m, 1))
                obj_func_coeff = np.append(obj_func_coeff, 0)
                constraints_coeff = np.append(constraints_coeff, unit_vec, axis=1)
            if op == Operator.GE:
                unit_vec = np.reshape(get_ith_unit_vector(i, m, -1), (m, 1))
                obj_func_coeff = np.append(obj_func_coeff, 0)
                constraints_coeff = np.append(constraints_coeff, unit_vec, axis=1)
        return obj_func_coeff, constraints_coeff

    def _add_artificial_variables(
            self,
            obj_func_coeff: ndarray,
            constraints_coeff: ndarray):
        m = len(constraints_coeff)
        n = len(obj_func_coeff)
        missing_unit_vector_idx_map = get_unit_vector_to_idx(m)
        for i in range(n):
            col = tuple(constraints_coeff[:, i])
            if col not in missing_unit_vector_idx_map:
                continue
            idx = missing_unit_vector_idx_map[col]
            del missing_unit_vector_idx_map[col]
        for unit_vec, idx in missing_unit_vector_idx_map.items():
            unit_vec = np.reshape(get_ith_unit_vector(idx, m), (m, 1))
            obj_func_coeff = np.append(obj_func_coeff, BigM(-1, 0))
            constraints_coeff = np.append(constraints_coeff, unit_vec, axis=1)
        return obj_func_coeff, constraints_coeff

    # TODO: 把 res_val 的 BigM 转为普通数字
    def solve(self):
        res_x, res_val = super().solve()
        if self.is_debug:
            return res_x, res_val
        res_x_artificial_part = res_x[self.artificial_var_idx:]
        no_sol_err = ValueError('This linear programming problem has no solution')
        for v in res_x_artificial_part:
            if abs(v) >= 1e-14:
                raise no_sol_err
        if is_big_m_like(res_val) and abs(res_val.a) >= 1e-14:
            raise no_sol_err
        return res_x[:self.slack_variable_idx], res_val


def case1():
    print('case1', end='\n\n')
    c = np.array([2, 3])
    a = np.array([
        [2, 2],
        [4, 0],
        [0, 5],
    ], dtype=np.float)
    b = np.array([12, 16, 15], dtype=np.float)
    operators = [Operator.LE, Operator.LE, Operator.LE]
    simplex = BigMMethod(c, a, b, operators, should_print_table=True)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)


def case2():
    print('case2', end='\n\n')
    c = np.array([-3, 0, 1])
    a = np.array([
        [1, 1, 1],
        [-2, 1, -1],
        [0, 3, 1],
    ], dtype=np.float)
    b = np.array([4, 1, 9], dtype=np.float)
    operators = [Operator.LE, Operator.GE, Operator.EQ]
    simplex = BigMMethod(c, a, b, operators, should_print_table=True)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)


def main():
    case1()
    case2()


if __name__ == '__main__':
    output_file_path = os.path.join('outp', 'big_m_method.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        sys.stdout = output_file
        main()
