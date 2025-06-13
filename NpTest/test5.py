import numpy as np

# data = np.arange(24).reshape(4, 3, 2)
# print(data)
# print(data[:, :, 0])
# print(np.take(data, [0], axis=2))
# print(np.take(data, []))
#
# arr = np.array([5, 10, 15, 20, 25])
# np.put(arr, [1, 3, 5], [99, 100, 101], mode='wrap')
# np.put(arr, [7, 8], -1, mode='wrap')
# print(arr)
# arr = np.zeros(10)
# arr[[2, 5, 8]] = [10, 20, 30]
# arr[1::2] = -1
# print(arr)
#
# data = np.arange(16).reshape(4, 4)
# print(data)
# print(data[[0, 1], [2, 3]])

data = np.ones((4, 4))
print(data)
data[::2, 1::2] = 0
data[1::2, ::2] = 0
print(data)