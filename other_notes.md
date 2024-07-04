```python
unit_vector_to_idx = dict([[np.hstack((np.zeros(i), 1, np.zeros(m - i - 1))), i] for i in range(m)])
```