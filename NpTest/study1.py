import numpy as np


# a = np.array([1, 2, 3])
# print(a + 6)
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr[:2, 1:])

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(a * b)
#
# arr = np.arange(9).reshape(3, 3)
# print(arr)
# print(np.mean(arr, axis=1))
#
# arr = np.array([1, 2, 3, 4, 5])
# print(np.mean(arr, axis=0))

a = np.array([1, 2, 3]).reshape(3, 1)
b = np.array([4, 5, 6])
a = np.hstack([a] * 3)
b = np.vstack([b] * 3)
print(a)
print(b)