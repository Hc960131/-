import numpy as np

# arr = np.arange(12).reshape(3, 4)
# print(arr[-1, :])
# print(arr[-2:, -2:])
# print(arr[-2:, -2:].flatten())

# arr = np.arange(16).reshape(4, 4)
# # print(arr[0::2, 1::2])
# # print(arr[1::2, 1::2])
# # print(arr[-1::-2, -1::-1])
# # print(np.flipud(arr))
# # print(arr[0::2, 1::2].flatten())
# # print(arr[::2, 1::2].ravel())
# b = arr[0::2, 1::2]
# c = b.flatten()
# d = b.ravel()
# b[0][1] = 100
# print(d)
# print(c)

# data = np.array([[10, 20], [30, 40]])
# mask = np.array([[True, False], [False, True]])
# print(data[mask])
# print(data)
# print(data > 25)
# print(data[np.eye(2, dtype=bool)])

# arr = np.zeros((3, 3))
# arr2 = arr
# arr3 = arr.copy()
# arr[[0, -1], :] = -1
# arr[:, [0, -1]] = -1
# print(arr)
# print(arr2)
# print(arr3)

arr = np.ones((3, 3, 3))
arr[:, [0, -1], :] = -1
arr[:, :, [0, -1]] = -1
print(arr)


