[TOC]

## 引言

TODO

环境：

- pytest 7.4.4
- numpy

## 单纯形法、大M法

解有三种情况：

1. 有唯一最优解
2. 有无穷多最优解
3. 有无界解

如何判断无界解：求出基变量时，如果发现系数矩阵的每一列都小于等于0，导致`theta = min(-, -, ...)`，那么线性规划问题具有无界解。

## 参考资料

1. https://www.geeksforgeeks.org/simplex-algorithm-tabular-method/
2. https://www.savemyexams.com/a-level/further-maths_decision-maths-1/edexcel/17/revision-notes/linear-programming/simplex-algorithm/big-m-method/
3. https://stackoverflow.com/questions/10253826/path-issue-with-pytest-importerror-no-module-named
4. https://pytest-with-eric.com/pytest-best-practices/pytest-ini/