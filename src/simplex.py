from typing import List, Optional
import sys
import os
import numpy as np
from tabulate import tabulate
from utils import get_unit_vector_to_idx, is_infinite


class Simplex:
    def __init__(
            self,
            obj_func_coeff: np.ndarray,
            constraints_coeff: np.ndarray,
            b: np.ndarray,
            should_print_table: bool = False,
            is_debug: bool = False) -> None:
        self.obj_func_coeff = obj_func_coeff
        # 保证外部数组不会被内部代码修改
        self.tmp_constraints_coeff = constraints_coeff.copy()
        self.constraints_coeff = constraints_coeff.copy()
        self.b = b
        self.should_print_table = should_print_table
        self.is_debug = is_debug
        self.n = len(self.obj_func_coeff)
        self.m = len(self.constraints_coeff)

    def _get_initial_basis(self) -> List[int]:
        m = self.m
        basis = [0] * m
        unit_vector_to_idx = get_unit_vector_to_idx(m)
        for i in range(self.n):
            col = tuple(self.tmp_constraints_coeff[:, i])
            if col not in unit_vector_to_idx:
                continue
            idx = unit_vector_to_idx[col]
            del unit_vector_to_idx[col]
            basis[idx] = i
        if unit_vector_to_idx:
            raise ValueError('Get initial basis failed')
        return basis

    def _spread_represent(self, basis: List[int], basis_sol) -> List:
        res = [0] * self.n
        for i, x in zip(basis, basis_sol):
            res[i] = x
        return res

    def _print_table(
            self,
            relative_profits: List,
            basis_sol: List,
            basis: List[int],
            cb: List,
            enter_basis: Optional[int] = None,
            leave_basis_vec_1_idx: Optional[int] = None):
        if not self.should_print_table:
            return

        m, n = self.m, self.n

        def get_constraints_coeff_to_display():
            return [
                [f'[{v}]' if i == leave_basis_vec_1_idx and j == enter_basis else v
                 for j, v in enumerate(constraints_row)]
                for i, constraints_row in enumerate(self.tmp_constraints_coeff)
            ]

        res = np.full((m + 3, n + 3), '', dtype='S20')
        res[0, 1] = 'cj'
        res[1, 0] = 'CB'
        res[1, 1] = 'Basis'
        res[1, 2] = 'b'
        res[1, 3:] = [f'x{v+1}' for v in range(n)]
        res[m + 2, 1] = 'sigma'
        res[2:m + 2, 0] = cb
        res[2:m + 2, 1] = [f'x{v+1}' for v in basis]
        res[2:m + 2, 2] = basis_sol
        res[0, 3:] = self.obj_func_coeff
        constraints_coeff_to_display = get_constraints_coeff_to_display()
        res[2:m + 2, 3:] = constraints_coeff_to_display
        res[m + 2, 3:] = relative_profits
        tbl = tabulate(res, tablefmt='orgtbl')
        print(tbl, end='\n\n')

    def solve(self):
        m, n = self.m, self.n
        self.tmp_constraints_coeff = self.constraints_coeff.copy()
        basis = self._get_initial_basis()
        basis_sol = self.b.tolist()
        cb = [self.obj_func_coeff[v] for v in basis]
        iteration_count = 0
        while True:
            relative_profits = [self.obj_func_coeff[i] - np.dot(self.tmp_constraints_coeff[:, i], cb) for i in range(n)]
            enter_basis = np.argmax(relative_profits)
            if relative_profits[enter_basis] <= 0:
                self._print_table(relative_profits, basis_sol, basis, cb)
                basis_sol_spread = self._spread_represent(basis, basis_sol)
                res_val = np.dot(basis_sol_spread, self.obj_func_coeff)
                return basis_sol_spread, res_val
            if iteration_count >= 100:
                raise RecursionError('This linear programming problem is not convergent')
            enter_basis_col = self.tmp_constraints_coeff[:, enter_basis]
            theta = np.divide(basis_sol, enter_basis_col, out=np.full(m, np.inf), where=enter_basis_col != 0)
            '''
            由于精度误差，可能会出现基变量值为0但被认为是负数，与负的系数矩阵元素一除得到正数的情况。
            因此系数矩阵元素必须为正的判定不可省。相关测试用例： test_big_m_case8
            '''
            theta = list(map(
                lambda it: it[1] if self.tmp_constraints_coeff[it[0], enter_basis] > 0 and it[1] >= 0 else float('inf'),
                enumerate(theta)
            ))
            leave_basis_vec_1_idx = np.argmin(theta)
            if is_infinite(theta[leave_basis_vec_1_idx]):
                raise ValueError('This linear programming problem is unbounded')
            self._print_table(relative_profits, basis_sol, basis, cb, enter_basis, leave_basis_vec_1_idx)
            main_ele_coeff = self.tmp_constraints_coeff[leave_basis_vec_1_idx, enter_basis]
            for i in range(m):
                if i == leave_basis_vec_1_idx:
                    continue
                cur_row_coeff = self.tmp_constraints_coeff[i, enter_basis]
                basis_sol[i] -= basis_sol[leave_basis_vec_1_idx] / main_ele_coeff * cur_row_coeff
                self.tmp_constraints_coeff[i, :] -= self.tmp_constraints_coeff[leave_basis_vec_1_idx, :] / main_ele_coeff * cur_row_coeff
            # 要么复制一下主元行再操作，要么先修改非主元行再修改主元行。我选后者
            # 这里没写 try except 是因为 theta[leave_basis_vec_1_idx] 有限就能保证主元不为0
            basis_sol[leave_basis_vec_1_idx] /= main_ele_coeff
            self.tmp_constraints_coeff[leave_basis_vec_1_idx, :] /= main_ele_coeff
            basis[leave_basis_vec_1_idx] = enter_basis
            cb = [self.obj_func_coeff[v] for v in basis]
            iteration_count += 1


def main():
    c = np.array([2, 3, 0, 0, 0])
    a = np.array([
        [2, 2, 1, 0, 0],
        [4, 0, 0, 1, 0],
        [0, 5, 0, 0, 1],
    ], dtype=np.float)
    b = np.array([12, 16, 15], dtype=np.float)
    simplex = Simplex(c, a, b, should_print_table=True)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)


if __name__ == '__main__':
    output_file_path = os.path.join('outp', 'simplex.txt')
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        sys.stdout = output_file
        main()
