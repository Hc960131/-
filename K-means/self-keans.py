import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# data = random.random((2, 2))
# data2 = random.random((3, 2))
# # print(data)
# # print(data[12:, :1])
# n_shape1 = data.shape[0]
# random_index = random.default_rng(10).permutation(n_shape1)[:3]
# # print(random.default_rng(10).permutation(n_shape1))
# # print(random_index)
# print(data[:, np.newaxis].shape)
# print(data[np.newaxis].shape)
# print(data)
# print(data2)
# print(data2 - data[:, np.newaxis])
# # print((data2 - data[:, np.newaxis]).sum(axis=0))
# # print((data2 - data[:, np.newaxis]).sum(axis=1))
# print((data2 - data[:, np.newaxis]).shape)
# print((data2 - data[:, np.newaxis]).sum(axis=2).shape)
# print(np.argmin((data2 - data[:, np.newaxis]).sum(axis=2), axis=0))

# data1 = np.random.random((3, 3, 2))
# data2 = np.random.random((3, 3, 2))
# print(data1.shape)
# # 对第一个维度进行堆叠
# data3 = np.vstack([data1, data2])
# # 对第二个维度进行堆叠
# data4 = np.hstack([data1, data2])
# print(data3.shape)
# print(data4.shape)
#
# print(np.random.randint(0, 10))

# data1 = np.random.random((3, 2))
# data2 = np.random.random((2, 2))
# print(np.allclose(data1, data2))


class Kmeans:
    def __init__(self, n_cluster, max_iter, random_state=42):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, x):
        assert isinstance(x, np.ndarray)
        n_shape = x.shape[0]
        random_idx = random.default_rng(self.random_state).permutation(n_shape)[:self.n_cluster]
        self.centroids = x[random_idx]

        for i in range(self.max_iter):
            distance = np.sqrt(((x - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distance, axis=0)

            # 这里有两种获取新中心的方法，我个人更推荐第一种，
            # 因为在第二种的情况下，如果某一个label没有对应的值，那么中心的shape会发生变化，导致后续np.allclose报错
            # 但是第一种考虑了这种情况，并做了冗余处理
            new_centers = self.get_new_center2(x)

            if np.allclose(new_centers, self.centroids):
                break
            self.centroids = new_centers

    def get_new_center(self, x):
        new_centers = np.zeros((self.n_cluster, x.shape[1]))
        for i in range(self.n_cluster):
            if np.sum(self.labels == i) > 0:
                new_centers[i] = x[self.labels == i].mean(axis=0)
            else:  # 如果某个簇没有点，则重新随机初始化
                new_centers[i] = x[np.random.randint(0, x.shape[0])]
        return new_centers

    def get_new_center2(self, x):
        return np.array([x[self.labels == k].mean(axis=0) for k in range(self.n_cluster)])

    def predict(self, x_predict):
        distance = np.sqrt(((x_predict - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distance, axis=0)


x_train = np.vstack([
    np.random.normal(loc=0, scale=1, size=(100, 2)),
    np.random.normal(loc=-2, scale=1, size=(100, 2)),
    np.random.normal(loc=2, scale=1, size=(100, 2))
])
x_test = np.vstack([
    np.random.normal(loc=0, scale=1, size=(20, 2)),
    np.random.normal(loc=-2, scale=1, size=(20, 2)),
    np.random.normal(loc=2, scale=1, size=(20, 2))
])
print(x_train.shape)
print(x_test.shape)
kmeans = Kmeans(n_cluster=3, max_iter=100)
kmeans.fit(x_train)
print(kmeans.predict(x_test))

