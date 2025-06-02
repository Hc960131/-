import numpy as np

# # np.concatenate，基于现有维度进行拼接，不会增加新的维度，除了拼接维度，其余维度必须一致
# a = np.array([[1, 2], [3, 4]])
# b = np.array([5, 6])
# print(b.shape)
# print(b[np.newaxis, :].shape)
# print(a.shape)
# print(np.concatenate((a, b[np.newaxis, :].T), axis=1))


# # np.stack 默认新增一个维度，np.stack堆叠，相当于先在指定堆叠的轴上增加一个维度，再进行np.concatenate
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(np.stack((a, b)))
# print(np.concatenate((a[np.newaxis, :], b[np.newaxis, :]), axis=0))
# print(np.concatenate((a[:, np.newaxis], b[:, np.newaxis]), axis=1))
# print(np.stack((a, b), axis=1))


# # np.vstack，行方向堆叠，对于一维数组，会直接增加一个维度，类似于a[np.newaxis, :]，对于二维数组，不会增加维度
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(np.concatenate((a[np.newaxis, :], b[np.newaxis, :]), axis=0))
# print(np.vstack((a, b)))

# # np.hstack，列方向进行堆叠，不会增加维度
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# print(np.hstack((a, b)))
# print(np.hstack((a[:, np.newaxis], b[:, np.newaxis])))

# # np.dstack，在第三个方向上进行堆叠，如果给定的数组维度不足3，则先补齐
# # 对于一维数组（n,）补齐完成之后是(1, n, 1)，相当于先a[np.newaxis, :][:, :, np.newaxis]，再np.concatenate，axis=2
# # 对于二维数组(m, n)，补齐完成之后是(m, n, 1)，相当于先m[:, :, np.newaxis]， ，再np.concatenate，axis=2
# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# # print(a[:, np.newaxis][:, np.newaxis])
# # print(a[np.newaxis, :])
# # print(a[np.newaxis, :].shape)
# # print(a[np.newaxis, :][:, :, np.newaxis].shape)
# print(np.dstack((a, b)))
# print(np.concatenate((a[np.newaxis, :][:, :, np.newaxis], b[np.newaxis, :][:, :, np.newaxis]), axis=2))
# # print(np.dstack((a[np.newaxis, :], b[np.newaxis, :])))
# # print(np.dstack((a[:, np.newaxis], b[:, np.newaxis])))
#
# m = np.arange(4).reshape(2, 2)
# print(m)
# n = np.linspace(5, 8, 4).reshape(2, 2)
# print(n)
# print(np.concatenate((m[:, :, np.newaxis], n[:, :, np.newaxis]), axis=2))
# print(np.dstack((m, n)))

# np.column_stack，按照列方向进行堆叠，对于一维数组，会先在列方向上增加维度，对于二维数组则不会改变维度
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.column_stack((a, b)))
print(np.hstack((a, b)))
print(np.column_stack((a[:, np.newaxis], b[:, np.newaxis])))
