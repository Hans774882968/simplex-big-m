[TOC]

# 手把手教你从零实现单纯形法的大M法，附若干线性规划模型应用题题解

## 引言

我大二下第一次看《运筹学基础及应用（第六版）》学解线性规划问题的单纯形法的人工变量法时，看到了这句话

> 用大M法处理人工变量，用手工计算求解时不会遇到麻烦。但用电子计算机求解时，对M就只能在计算机内输入一个机器最大字长的数字。

我实在不能苟同，因为只要写一个表示大M的结构体，然后模拟手工计算的过程，就不会有这个问题。于是我写了[这个项目](https://github.com/Hans774882968/simplex-big-m)。在写代码前，我搜了几个其他的项目，发现代码量都很大，而且~~一个都看不懂~~，索性还是自己从头写了。

环境：

- Python 3.7.6
- pytest 7.4.4
- numpy 1.18.1

**作者：[hans774882968](https://blog.csdn.net/hans774882968)以及[hans774882968](https://juejin.cn/user/1464964842528888)以及[hans774882968](https://www.52pojie.cn/home.php?mod=space&uid=1906177)**

本文52pojie：https://www.52pojie.cn/thread-1943754-1-1.html

本文juejin：https://juejin.cn/post/7390957334425600052

本文CSDN：https://blog.csdn.net/hans774882968/article/details/140408547

## 单纯形法、大M法，及相关线性代数知识

单纯形法是求解一般线性规划问题的迭代算法。其基本思想是从一个初始的基可行解出发，通过迭代逐步逼近最优解。每一步迭代都选择一个新的基可行解，使得目标函数值逐步改善，直到达到最优解。

因为看其他资料可知，线性规划问题可以约定其他的标准形式，所以本文和《运筹学基础及应用（第六版）》一样，约定线性规划问题的标准形式为

```
max CX
AX = b
X >= 0
C: 1 * n, X: n * 1, A: m * n, b: m * 1, 一般认为 m <= n
```

在我的代码中，C叫`obj_func_coeff`，A叫`constraints_coeff`，b叫the RHS of the constraints。

### 预备知识

- 凸集：如果一个集合中的任意两点的连线都完全包含在该集合内，那么这个集合就是一个凸集。换句话说，对于集合 ( C ) 中的任意两点`x`和`y`，以及任意`\lambda`满足`0 \leq \lambda \leq 1`，点`\lambda x + (1 - \lambda) y`也在集合 ( C ) 中。
- 顶点：在多面体（例如多边形或多面体）中，顶点是多面体的角点。对于线性规划问题，顶点是可行区域的边界上的点，这些点是由约束条件的交点形成的。代数定义：任意`X1, X2 ∈ C`，不存在`X = a * X1 + (1-a) * X2, 0 < a < 1`。

以上概念主要出现在讨论线性规划问题的可行域时。

- 基：在线性代数中，基是一个向量空间中的一组向量，这些向量线性无关且能够生成整个向量空间。在线性规划中，基是指约束条件矩阵中的一组线性无关的列向量。为什么基很重要？因为**单位向量组是最简单的一组基**，通过单位向量组我们立刻可以得到初始基可行解，从而开始运行单纯形法。值得注意的是，（1）`m`个`m`维向量线性无关/每个向量都不能由其他`m-1`个向量线性表出。（2）它们按列或按行组成的矩阵可逆/行列式不为0/满秩。都是对同一个事物的不同角度的描述。我们在系数矩阵的列向量组中选出一组基，则这些列向量对应的变量叫做**基变量**，其他变量叫做非基变量。
- 基解：令`n - m`个非基变量都为0，根据上一段的（1）和（2）（或者按《运筹学基础及应用（第六版）》的说法“根据克拉默规则”），基变量可得唯一解。于是一组基对应一个基解。当选择的基为单位向量组时，我们立刻可以得到初始基可行解，从而开始运行单纯形法。
- 可行解：满足所有约束条件的解就是可行解。所有可行解组成的集合叫可行域。
- 基可行解：基解和可行解的交集。
- 可行基：可行基是指对应于基可行解的基。

### 单纯形法的计算步骤

在单纯形法的计算步骤中，主要是需要计算检验数（Relative Profits）和`theta`，它们的公式的推导过程分别在《运筹学基础及应用（第六版）》第一章 3-5 最优性检验和解的判别和 3-4 从初始基可行解转换为另一基可行解中有详细说明。不理解也没事，把式子背下来就足够应付考试了。计算步骤总结如下：

1. **标准化问题**：将线性规划问题转化为标准形式，即目标函数为最大化形式，约束条件为等式形式，且所有变量非负。这一步需要加上若干松弛变量（Slack Variables）。
2. **构造初始单纯形表**：找到一个初始的基可行解，并构造相应的单纯形表。在大M法中再具体讨论。
3. **判断最优解**：如果单纯形表中所有检验数都≤0，则当前单纯形表中的基可行解就是最优解；否则继续迭代。
4. **迭代计算**：从一个基可行解转换到另一个更优秀的基可行解。（1）确定入基变量。首先算`n`个检验数：`sigma[j] = C[j] - C_B * A[:, j]`。显然基变量对应的检验数都是0。每个大于0的检验数都可以作为入基变量，但是我们一般会贪心地选择**检验数最大**的变量，因为这样似乎可以更快收敛。（2）确定出基变量。假设上一步确定入基变量是`xk`，那么`theta = min(bi / A[i, k], A[i, k] > 0)`。最小`theta`对应的每个`i`都可以作为出基变量。假设其中一个是`p`，`A[p, k]`就是下一步初等行变换的主元，我们会在单纯形表中把主元框起来。（3）更新单纯形表。我们要进行初等行变换，把主元列变成单位列向量。

文字说明看不懂也没事，可以结合下面的例题理解。

### 例题：《运筹学基础及应用（第六版）》第一章例5

```python
max z = 2*x1 + 3*x2
2*x1 + 2*x2 <= 12
4*x1 <= 16
5*x2 <= 15
x1, x2 >= 0
```

懒得画精美的markdown表格了，直接展示我代码输出的单纯形表吧！

- CB：基变量`xj`对应的`C[j]`，写出来是为了方便算内积，求出检验数。
- b：当前基可行解。
- σ：检验数。
- 被框起来的元素：初等行变换的主元。所在的行对应出基变量，所在的列对应入基变量。

```
|    | cj    |    | 2 | 3   | 0 | 0 | 0 |
| CB | Basis | b  | x1| x2  | x3| x4| x5|
| 0  | x3    | 12 | 2 | 2   | 1 | 0 | 0 |
| 0  | x4    | 16 | 4 | 0   | 0 | 1 | 0 |
| 0  | x5    | 15 | 0 | [5] | 0 | 0 | 1 |
|    | sigma |    | 2 | 3   | 0 | 0 | 0 |

|    | cj    |    | 2   | 3 | 0 | 0 | 0      |
| CB | Basis | b  | x1  | x2| x3| x4| x5     |
| 0  | x3    | 6  | [2] | 0 | 1 | 0 | -2 / 5 |
| 0  | x4    | 16 | 4   | 0 | 0 | 1 | 0      |
| 3  | x2    | 3  | 0   | 1 | 0 | 0 | 1 / 5  |
|    | sigma |    | 2   | 0 | 0 | 0 | -3 / 5 |

|    | cj    |   | 2 | 3 | 0     | 0 | 0      |
| CB | Basis | b | x1| x2| x3    | x4| x5     |
| 2  | x1    | 3 | 1 | 0 | 1 / 2 | 0 | -1 / 5 |
| 0  | x4    | 4 | 0 | 0 | -2    | 1 | 0.8    |
| 3  | x2    | 3 | 0 | 1 | 0     | 0 | 0.2    |
|    | sigma |   | 0 | 0 | -1    | 0 | -1 / 5 |
```

步骤：

1. 选择`x2`为入基变量，计算`theta = min(12 / 2, -, 15 / 5)`确定`x5`为出基变量。初等行变换：`(1) -= 2/5*(3), (2) = (2), (3) /= 5`。
2. 选择`x1`为入基变量，计算`theta = min(6 / 2, 16 / 4, -)`确定`x3`为出基变量。初等行变换：`(1) /= 2, (2) -= 4/2*(1), (3) = (3)`。

3. 所有检验数都小于0，没有等于0的，所以有唯一最优解`x1 = 3, x2 = 3, x4 = 3, x3 = x5 = 0`，目标函数值15。


### 解的判别

解有几种情况：

1. 有唯一最优解
2. 有无穷多最优解
3. 有无界解
4. 无解

如何判断无穷多最优解：我们知道求得最优解时，所有变量的检验数都小于等于0。如果存在一个非基变量的检验数等于0，并且选择它为入基变量时可以正常求出`theta`，那么就能正常求出出基变量+进行初等行变换，于是就找到了另一个顶点也是最优解。于是这两个顶点连线上的所有点都是最优解。

如何判断无界解：求出基变量时，如果发现系数矩阵的每一列都小于等于0，导致`theta = min(-, -, ...)`，那么线性规划问题具有无界解。

### 人工变量法（大M法）

化为标准形式后，如果系数矩阵中不存在单位矩阵，那么我们需要先写代码找到一组基，找到基后得到的基解还不一定是可行解。所以我们有一个很朴素的想法，就是人工地添加所有缺失的单位列向量。为此，对应新增的人工变量（Artificial Variables）的系数要设为负无穷大，用`-M`表示，以保证解出的人工变量取值为0。建立初始单纯形表后，正常运行单纯形法求解即可。

比如《运筹学基础及应用（第六版）》第一章例6

```python
max z = -3 * x1 + x3
x1 + x2 + x3 + x4 = 4
-2 * x1 + x2 - x3 - x5 = 1
3 * x2 + x3 = 9
x1~x5 >= 0
```

其列向量

```python
P1 P2 P3 P4 P5
1  1  1  1  0
-2 1  -1 0  -1
0  3  1  0  0
```

可见只有一个单位列向量`P4`。那么我们再加两个单位列向量`P6, P7`

```python
P1 P2 P3 P4 P5 P6 P7
1  1  1  1  0  0  0
-2 1  -1 0  -1 1  0
0  3  1  0  0  0  1
```

目标函数相应地变为`max z = -3 * x1 + x3 - M * x6 - M * x7`。

## 核心代码讲解

[项目传送门](https://github.com/Hans774882968/simplex-big-m)。目录结构：

```
SIMPLEX-BIG-M
│  .gitignore
│  pytest.ini
│  README.md
│
├─outp
│      big_m_method.txt
│      main.txt
│      simplex.txt
│
├─src
│  │  big_m.py
│  │  big_m_method.py
│  │  consts.py
│  │  main.py
│  │  simplex.py
│  │  utils.py
│  └─ __init__.py
│
└─test
    │  test_big_m.py
    │  test_big_m_method.py
    │  test_simplex.py
    │  test_utils.py
    └─ __init__.py
```

开发过程主要分两阶段。

### 第一阶段：实现`BigM`类和`Simplex`类

`BigM`类就像复数一样，支持各种符号运算及与浮点数的混合运算。`Simplex`类支持输入一个列向量组含有所有单位向量的矩阵（行数`m`≤列数`n`），首先求出初始基可行解，然后实现单纯形法的各步。`Simplex`类应支持目标函数系数含有`BigM`类实例。

`BigM`类构造函数：

```python
class BigM:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
```

`BigM`类要支持的运算符不少，所以代码量略大，[完整代码](https://github.com/Hans774882968/simplex-big-m/blob/main/src/big_m.py)。为了减少代码量，一些运算符可以由其他运算符实现。比如只实现＜和=即可实现其他4个比较运算符≤、＞、≥、≠。又比如`__radd__, __rmul__`可以由`__add__, __mul__`实现。接下来看一些运算符的实现。＜：

```python
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
```

这里用`is_big_m_like(other)`而非`isinstance(other, BigM)`的原因是单测文件导入的`BigM`位于`sys.modules['src/big_m']`，而`src`文件夹下其他文件导入的`BigM`位于`sys.modules['big_m']`，两者不一样，于是我写了`is_big_m_like(other)`暂时规避了这个问题。

乘法：可类比复数类的实现。

```python
    def __mul__(self, other: object):
        if isinstance(other, (int, float)):
            return BigM(self.a * other, self.b * other)
        if not is_big_m_like(other):
            raise ValueError(f'Expect a BigM instance, but got type {type(other)}')
        return BigM(self.a * other.a + self.b * other.a + self.a * other.b, self.b * other.b)
```

[完整代码](https://github.com/Hans774882968/simplex-big-m/blob/main/src/big_m.py)，[BigM类单测](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m.py)。

接下来看最难的部分——单纯形法的实现。[Simplex类完整代码](https://github.com/Hans774882968/simplex-big-m/blob/main/src/simplex.py)，[Simplex类单测](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_simplex.py)。

```python
class Simplex:
    def __init__(
            self,
            obj_func_coeff: np.ndarray,
            constraints_coeff: np.ndarray,
            b: np.ndarray,
            should_print_table: bool = False,
            is_debug: bool = False) -> None:
        self.obj_func_coeff = obj_func_coeff  # 目标函数向量
        # 保证外部数组不会被内部代码修改
        self.tmp_constraints_coeff = constraints_coeff.copy()  # 会被修改的约束系数矩阵
        self.constraints_coeff = constraints_coeff.copy()  # 约束系数矩阵
        self.b = b  # 约束右侧向量，the RHS of the constraints
        self.should_print_table = should_print_table  # 是否打印各步的单纯形表
        self.is_debug = is_debug  # 单纯为了方便测试
        self.n = len(self.obj_func_coeff)
        self.m = len(self.constraints_coeff)
        self._check_problem_shape()
```

通过单位列向量求初始基解：

```python
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

    def solve(self):
        m, n = self.m, self.n
        self.tmp_constraints_coeff = self.constraints_coeff.copy()
        basis = self._get_initial_basis()
        basis_sol = self.b.tolist()
        cb = [self.obj_func_coeff[v] for v in basis]
```

求检验数和入基变量：求内积即可。

```python
            relative_profits = [self.obj_func_coeff[i] - np.dot(self.tmp_constraints_coeff[:, i], cb) for i in range(n)]
            enter_basis = np.argmax(relative_profits)
```

成功条件：`if relative_profits[enter_basis] <= 0`。

求`theta`和出基变量：

```python
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
```

最开始实现时我认为不需要特意判断系数矩阵元素是否为正，但是精度误差可能会导致0变为正数或负数，导致bug出现，因此还是无奈地加上了判定逻辑。另外，为了防止出现`RuntimeWarning`，用了`np.divide`而非`/`来做除法。

在单纯形表中，被框住的元素就是主元。我们要进行初等行变换，把主元列变成单位列向量：

```python
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
```

和实现高斯消元法时一样，要注意修改主元行和非主元行的顺序。

接下来看看解的判定逻辑。之前说过线性规划问题的解有以下情况：无解、唯一解、无穷多解、无界解。~~因为我胸无大志所以~~就不判断无穷多解的情况了。无界解：在上面求`theta`和出基变量的代码里，我们用`np.inf`来表示所有被跳过的列，所以如果最后得到的`theta`每一列都是正无穷，那么原问题就是无界的。

```python
from utils import get_unit_vector_to_idx, is_infinite
# ...
            if is_infinite(theta[leave_basis_vec_1_idx]):
                raise ValueError('This linear programming problem is unbounded')
```

无解的情况放在`BigMMethod`类实现。最后还有一个兜底逻辑，如果因为“相持”（详见《运筹学基础及应用（第六版）》第一章“5-4 单纯形法计算中的相持”一节）等原因导致算法陷入死循环，我们就报个错。

```python
    def solve(self):
        # ...
        iteration_count = 0
        while True:
            # ...
            if iteration_count >= 100:
                raise RecursionError('This linear programming problem is not convergent')
            # ...
            iteration_count += 1
```

最后再做件锦上添花的事：打印单纯形表。选择的包：`from tabulate import tabulate`。用法：

```python
        tbl = tabulate(res, tablefmt='orgtbl')
        print(tbl, end='\n\n')
```

`res`是一个可迭代的二维数组即可，因此为了方便，我们可以直接用`np.ndarray`。因为numpy主要是C实现的，所以初始化这个字符串二维数组需要指明`char[]`长度：`res = np.full((m + 3, n + 3), '', dtype='S20')`。

代码比较长，不贴了，方法名是`_print_table`，感兴趣的同学可以去我的项目找。推荐的调用时机：每次循环进行初等行变换前（即修改系数矩阵前的最后时刻）、成功得到解时。输出的单纯形表示例：

```
|    | cj    |      | 2   | 3     | 0   | 0   | 0   |
| CB | Basis | b    | x1  | x2    | x3  | x4  | x5  |
| 0  | x3    | 12.0 | 2.0 | 2.0   | 1.0 | 0.0 | 0.0 |
| 0  | x4    | 16.0 | 4.0 | 0.0   | 0.0 | 1.0 | 0.0 |
| 0  | x5    | 15.0 | 0.0 | [5.0] | 0.0 | 0.0 | 1.0 |
|    | sigma |      | 2.0 | 3.0   | 0.0 | 0.0 | 0.0 |

|    | cj    |      | 2     | 3   | 0   | 0   | 0                   |
| CB | Basis | b    | x1    | x2  | x3  | x4  | x5                  |
| 0  | x3    | 6.0  | [2.0] | 0.0 | 1.0 | 0.0 | -0.4                |
| 0  | x4    | 16.0 | 4.0   | 0.0 | 0.0 | 1.0 | 0.0                 |
| 3  | x2    | 3.0  | 0.0   | 1.0 | 0.0 | 0.0 | 0.2                 |
|    | sigma |      | 2.0   | 0.0 | 0.0 | 0.0 | -0.6000000000000001 |

|    | cj    |     | 2   | 3   | 0    | 0   | 0                    |
| CB | Basis | b   | x1  | x2  | x3   | x4  | x5                   |
| 2  | x1    | 3.0 | 1.0 | 0.0 | 0.5  | 0.0 | -0.2                 |
| 0  | x4    | 4.0 | 0.0 | 0.0 | -2.0 | 1.0 | 0.8                  |
| 3  | x2    | 3.0 | 0.0 | 1.0 | 0.0  | 0.0 | 0.2                  |
|    | sigma |     | 0.0 | 0.0 | -1.0 | 0.0 | -0.20000000000000007 |
```

### 第二阶段：通过增强`Simplex`类实现`BigMMethod`类

人工变量法就是单纯形法的增强，因此从直觉上看设计为类继承很合理。

```python
class BigMMethod(Simplex):
    def __init__(
            self,
            obj_func_coeff: ndarray,
            constraints_coeff: ndarray,
            b: ndarray,
            operators: List[str],
            should_print_table: bool = False,
            is_debug: bool = False) -> None:  # is_debug: 单纯为了方便测试
        self.operators = operators  # 支持 <= = >= 三种运算符
        self.slack_variable_idx = len(obj_func_coeff)
        obj_func_coeff, constraints_coeff = self._to_standard_model(obj_func_coeff, constraints_coeff, operators)
        self.artificial_var_idx = len(obj_func_coeff)
        obj_func_coeff, constraints_coeff = self._add_artificial_variables(obj_func_coeff, constraints_coeff)
        super().__init__(obj_func_coeff, constraints_coeff, b, should_print_table, is_debug)
        self._check_problem_shape()

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
```

先化为标准问题（`_to_standard_model`），再看缺失了哪些单位向量，补上对应的人工变量（`_add_artificial_variables`），然后调用父类的初始化，最后和父类一样检查一下各个矩阵的形状。

`solve`方法：先调用父类的`solve`，再进行一些后置检查即可。

```python
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
```

[完整代码](https://github.com/Hans774882968/simplex-big-m/blob/main/src/big_m_method.py)，[单测](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m_method.py)。为了严谨，找了不少用例，所以代码量看上去大。

## 单测

假设目录结构如下：

```
SIMPLEX-BIG-M
│  pytest.ini
│
├─src
│  │  big_m.py
│  │  utils.py import了big_m.py
│  └─ __init__.py
│
└─test
    │  test_utils.py
    └─ __init__.py
```

那么用`pytest --html=coverage/report.html`命令运行单测时会在utils.py报错`import big_m`失败。根据[参考链接3](https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named)、[参考链接4](https://pytest-with-eric.com/pytest-best-practices/pytest-ini/)，需要新增`pytest.ini`：

```ini
[pytest]
pythonpath = . src
```

## 线性规划模型应用题收集

### 《运筹学基础及应用（第六版）》P60 习题1.22

> 要制作100套钢筋架子，每套有长2.9m、2.1m和1.5m的钢筋各一根。已知原材料长7.4m，应如何切割，使用原材料最节省，试建立线性规划模型并求解。

先枚举所有切割方案：`(0, 0, 4), (0, 1, 3), (0, 2, 2), (0, 3, 0), (1, 0, 3), (1, 1, 1), (1, 2, 0), (2, 0, 1)`，然后设每种切割方案使用的根数为`xi`，于是得到8变量，3约束条件的线性规划问题。目标函数显然是`min sum(xi, 1 <= i <= 8)`。[代码](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m_method.py)：

```python
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
```

### 《运筹学基础及应用（第六版）》P58 习题1.14

> 某厂在今后四个月内需租用仓库堆放物资。已知各月份所需仓库面积数字列于表1-28。仓库租借费用随合同期定，期限越长折扣越大，具体数字见表1-29。租用仓库的合同每月初都可办理，每份合同具体规定租用面积数和期限。因此该厂可根据需要，在任何一个月初办理租借合同。每次办理时可签一份，也可签若干份租用面积和租用期限不同的合同。问如何安排合同使得总共所付的租金最少？
>
> 表1-28
>
> | 月份                  | 1    | 2    | 3    | 4    |
> | --------------------- | ---- | ---- | ---- | ---- |
> | 所需仓库面积（100平） | 15   | 10   | 20   | 12   |
>
> 表1-29
>
> | 合同租借期限               | 1个月 | 2个月 | 3个月 | 4个月 |
> | -------------------------- | ----- | ----- | ----- | ----- |
> | 合同期内的租费（元/100平） | 2800  | 4500  | 6000  | 7300  |

设第`i`个月租`j`个月的合同的租借面积为`xij`，则有`i+j<=5`，所以共有10个变量。

目标函数：`min(2800 * sum(xi1, i = 1~4) + 4500 * sum(xi2, i = 1~3) + 6000 * sum(xi3, i = 1~2) + 7300 * x14)`。

约束条件：能覆盖到第`i`个月的合同的总面积大于等于所需仓库面积。

```python
x11+x12+x13+x14 >= 15
x12+x13+x14+x21+x22+x23 >= 10
x13+x14+x22+x23+x31+x32 >= 20
x14+x23+x32+x41 >= 12
```

据此填出约束条件系数矩阵，发现比较优美，分别由`4*4~1*1`的上三角阵构成。

```python
a = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
], dtype=np.float)
```

解得`x11 = 5, x14 = 10, x31 = 8, x32 = 2`，这表示第1个月租1个月合同500平，租4个月合同1000平，第3个月租1个月合同800平，租2个月合同200平。于是第1~4个月分别覆盖到500 + 1000，1000，1000 + 800 + 200，1000 + 200平，恰好都等于每个月所需仓库面积。[代码](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m_method.py)：

```python
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
```

### 《运筹学基础及应用（第六版）》P58 习题1.16

>某厂生产三种产品Ⅰ、Ⅱ、Ⅲ。每种产品要经过A、B两道工序加工。设该厂有两种规格的设备能完成A工序，以A1、A2表示；有三种规格的设备能完成B工序，以B1、B2、B3表示。产品Ⅰ可在A、B任何一种规格设备上加工；产品Ⅱ可在任何规格的A设备上加工，但完成B工序时，只能在B1设备上加工；产品Ⅲ只能在A2与B2设备上加工。已知在各种机床设备的单件工时、原材料费、产品销售价格、各种设备有效台时以及满负荷操作时机床设备的费用如表1-30所列，试安排最优的生产计划，使该厂利润最大。
>
>| 设备             | 产品Ⅰ | 产品Ⅱ | 产品Ⅲ | 设备有效台时 | 满负荷时的设备费用/元 |
>| ---------------- | ----- | ----- | ----- | ------------ | --------------------- |
>| A1               | 5     | 10    |       | 6000         | 300                   |
>| A2               | 7     | 9     | 12    | 10000        | 321                   |
>| B1               | 6     | 8     |       | 4000         | 250                   |
>| B2               | 4     |       | 11    | 7000         | 783                   |
>| B3               | 7     |       |       | 4000         | 200                   |
>| 原料费/（元/件） | 0.25  | 0.35  | 0.50  |              |                       |
>| 单价/（元/件）   | 1.25  | 2.00  | 2.80  |              |                       |

这题至少有两个做法。书上这题答案的思路是我下面提供的方法二，但给到的答案是方法一的，只能说书作者抄别人的书都抄不明白…

方法一：来自[参考链接5](https://blog.csdn.net/weixin_45755831/article/details/113216833)。数据是完全一样的，所以我在代码里设了一个`pc`数组（pc = process costing）

```python
pc = [0.05, 0.0321, 0.0625, 783 / 7000, 0.05]
```

设产品1经过A工序的数量为`x1~x3`，经过B工序的数量为`x4~x5`，则有`x1+x2+x3=x4+x5`。同理设`x6~x10`，有`x6+x7=x8`，`x9=x10`。

目标函数：获利=(售价-原料费)*产品总数-设备加工费。所以有

```python
max((x1 + x2) + 1.65 * x8 + 2.3 * x9 - pc[0] * 5 * x1 - pc[1] * 7 * x2 - pc[2] * 6 * x3 - pc[3] * 4 * x4 - pc[4] * 7 * x5 - pc[0] * 10 * x6 - pc[1] * 9 * x7 - pc[2] * 8 * x8 - pc[1] * 12 * x9 - pc[3] * 11 * x10)
```

约束条件：（1）不超过设备有效台时。（2）上面3个等式。

```python
5 * x1 + 10 * x6 <= 6000
7 * x2 + 9 * x7 + 12 * x9 <= 10000
6 * x3 + 8 * x8 <= 4000
4 * x4 + 11 * x10 <= 7000
7 * x5 <= 4000
```

在下面的代码里，我把`x10`都换成`x9`，减少一个变量。

最初的解：`[1200.0, 230.0492610837435, 0, 858.620689655172, 571.4285714285714, 0, 500.0, 500.0, 324.1379310344829] 1146.5665024630541`。但原问题是一个整数规划问题，于是我多尝试运行了几次，最后手工加了3条限制条件，成功缚住苍龙！

方法二：生产产品1共2*3种方案，设经A1、B1加工的产品1数量为`x11`，同理A1、B2到A2、B3的产品数量为`x12~x16`。同理设`x21, x22, x3`。设方法一的变量以y为开头，则有`x11+x12+x13=y1`，`x14+x15+x16=y2`，`x11+x14=y3`，`x12+x15=y4`等。理论上把这些等式代入方法一即可得到目标函数和约束条件。

目标函数和方法一同理推导，但是比方法一的式子要长一些，在此就不完整写出来了。仅以设备A2的加工费为例：`pc[1] * 7 * x14 + pc[1] * 9 * x22 + pc[1] * 12 * x3`。

和方法一不同，方法二的约束条件只有一种：不超过设备有效台时。和方法一同理推导即可，不完整列出了，仅以设备B1为例：`6 * (x11 + x14) + 8 * (x21 + x22) <= 4000`。

最初的解：`[0, 858.620689655172, 341.37931034482796, 0, 0, 230.0492610837435, 0, 500.0, 324.1379310344829] 1146.5665024630541`。后面我同样加了3条限制条件，得最终解。

[完整代码](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m_method.py)：

```python
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
```

### 《运筹学基础及应用（第六版）》P59 习题1.21

>北海银行一个分理处每天各时段对职员的需求如表1-32所示：
>
>| 时段     | 9~10 | 10~11 | 11~12 | 12~13 | 13~14 | 14~15 | 15~16 | 16~17 |
>| -------- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
>| 所需人数 | 4    | 5     | 6     | 6     | 5     | 6     | 8     | 8     |
>
>该分理处分别聘用部分全日制和部分非全日制职员。全日制职员每天从9:00工作到17:00，中间安排一小时午休（分两批，一批为`12:00~13:00`，另一批为`13:00~14:00`），每天薪金240元。非全日制职员分六批次上班，时间分别是`9:00~12:00`，`10：00~13:00`，`11:00~14:00`，`12:00~15:00`，`13:00~16:00`，`14:00~17:00`，每人每天薪金80元。问该分理处应聘用全日制和各批次的非全日制职员各多少人，能满足需求又使得薪金支出为最少？

设全日制员工12到13点用餐、13到14点用餐的数量分别为`x1, x2`，各个批次的非全日制员工数量分别为`y1~y6`。目标函数显然。约束的系数矩阵按列填写即可做到快速填写。

书上答案给的`res_x = [3, 3, 0, 0, 3, 0, 0, 2], res_val = -1840`是错的，答案应该是`res_x = [0, 0, 4, 1, 5, 0, 0, 8], res_val = -1440`。[代码](https://github.com/Hans774882968/simplex-big-m/blob/main/test/test_big_m_method.py)：

```python
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
```

## 参考资料

1. https://www.geeksforgeeks.org/simplex-algorithm-tabular-method/
2. https://www.savemyexams.com/a-level/further-maths_decision-maths-1/edexcel/17/revision-notes/linear-programming/simplex-algorithm/big-m-method/
3. https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named
4. https://pytest-with-eric.com/pytest-best-practices/pytest-ini/
5. 《运筹学基础及应用（第六版）》P58 习题1.16方法一：https://blog.csdn.net/weixin_45755831/article/details/113216833