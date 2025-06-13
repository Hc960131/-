import numpy as np

data = np.array([5, 10, 15, 20, 25, 30])
print(data[[1, 4, 5]])
print(data[1::2])

arr = np.random.randint(0, 100, 10)
print(arr[(arr >50) & (arr % 2 != 0)])
arr[arr < 30] = -1

# 给定4x4矩阵，交换其第1列和第3列
matrix = np.arange(16).reshape(4,4)
# 你的代码：
matrix[:, [0,2]] = matrix[:, [2,0]]
print(matrix)