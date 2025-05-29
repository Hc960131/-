from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, adjusted_rand_score

x, y = make_blobs(n_samples=300, n_features=2, cluster_std=1, random_state=10)
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)
kmeans = KMeans(n_clusters=3)
kmeans.fit(x_train)
y_predict = kmeans.predict(x_test)
print(y_test)
print(y_predict)
print(accuracy_score(y_test, y_predict))
# 关于为什么使用下面这个准确率计算方式，我写了个博客
# https://blog.csdn.net/h3076817064/article/details/148320313?sharetype=blogdetail&sharerId=148320313&sharerefer=PC&sharesource=h3076817064&spm=1011.2480.3001.8118
print(adjusted_rand_score(y_test, y_predict))


