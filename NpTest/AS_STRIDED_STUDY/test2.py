import numpy as np

from test1 import sequence_stride


# arr = np.arange(100)
# print(arr)
# # 计算每三个数的平均值
# result = sequence_stride(arr, 3)
# print(np.mean(result, axis=1))

arr2 = np.random.randn(100)
result2 = sequence_stride(arr2, 2)
print(arr2)
print(np.max(result2, axis=1))