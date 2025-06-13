import numpy as np

# arr = np.arange(20).reshape(4, 5)
# print(arr[1, [0, 2, 4]])
# mask = arr % 3 == 0
# print(arr[mask])
#
# arr[0, :] = arr[3, :][::-1]
# print(arr)

# arr = np.arange(36).reshape(6, 6)
# print(arr[2: 4, 2: 4])
# print(arr[::-1, ::-1])
# arr[0::2, 0::2] = 0
# print(arr)

# data = np.random.randint(0, 10, (5, 5))
# print(data[np.logical_and(data > 5, data % 2 != 0)])
# # print(data[np.eye(5, 5, dtype=bool)])
# print(np.max(data, axis=1))
# print(np.max(data, axis=1, keepdims=True))
# print(data - np.max(data, axis=1).reshape(5, 1))

arr = np.zeros((6, 6))
arr[2, 2] = 1
arr[[0, -1], :] = 1
arr[:, [0, -1]] = 1
print(arr)

print(arr[::2])

arr[::2], arr[1::2] = arr[1::2], arr[::2].copy()
print(arr)
