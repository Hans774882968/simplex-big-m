from typing import List
import numpy as np


class Simplex:
    def __init__(self, obj_func_coeff: np.ndarray, constraints_coeff: np.ndarray, b: np.ndarray) -> None:
        self.obj_func_coeff = obj_func_coeff
        self.constraints_coeff = constraints_coeff
        self.b = b
        self.n = len(self.obj_func_coeff)
        self.m = len(self.constraints_coeff)

    def _get_initial_basis(self) -> List[int]:
        m = self.m
        basis = [0] * m
        unit_vector_to_idx = dict([[tuple([0] * i + [1] + [0] * (m - i - 1)), i] for i in range(m)])
        for i in range(self.n):
            col = tuple(self.constraints_coeff[:, i])
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

    def solve(self):
        # TODO: 打印每一步的单纯形表
        n, m = self.n, self.m
        basis = self._get_initial_basis()
        basis_sol = self.b.tolist()
        cb = [self.obj_func_coeff[v] for v in basis]
        iteration_count = 0
        while True:
            relative_profits = [self.obj_func_coeff[i] - np.dot(self.constraints_coeff[:, i], cb) for i in range(n)]
            enter_basis = np.argmax(relative_profits)
            if relative_profits[enter_basis] <= 0:
                basis_sol_spread = self._spread_represent(basis, basis_sol)
                res_val = np.dot(basis_sol_spread, self.obj_func_coeff)
                return basis_sol_spread, res_val
            if iteration_count >= 100:
                raise RecursionError('This linear programming problem is not convergent')
            enter_basis_col = self.constraints_coeff[:, enter_basis]
            theta = np.divide(basis_sol, enter_basis_col, out=np.full(m, np.inf), where=enter_basis_col != 0)
            theta = list(map(lambda v: v if v >= 0 else float('inf'), theta))
            leave_basis_vec_1_idx = np.argmin(theta)
            leave_basis = basis[leave_basis_vec_1_idx]
            main_ele_coeff = self.constraints_coeff[leave_basis_vec_1_idx, enter_basis]
            for i in range(m):
                if i == leave_basis_vec_1_idx:
                    continue
                cur_row_coeff = self.constraints_coeff[i, enter_basis]
                basis_sol[i] -= basis_sol[leave_basis_vec_1_idx] / main_ele_coeff * cur_row_coeff
                self.constraints_coeff[i, :] -= self.constraints_coeff[leave_basis_vec_1_idx, :] / main_ele_coeff * cur_row_coeff
            # 要么复制一下主元行再操作，要么先修改非主元行再修改主元行。我选后者
            basis_sol[leave_basis_vec_1_idx] /= main_ele_coeff
            self.constraints_coeff[leave_basis_vec_1_idx, :] /= main_ele_coeff
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
    simplex = Simplex(c, a, b)
    res_x, res_val = simplex.solve()
    print(res_x, res_val)


if __name__ == '__main__':
    main()
