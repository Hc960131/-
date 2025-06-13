import numpy as np

# a = np.arange(32).reshape(4, 2, 4)
# b = np.arange(2).reshape(2, 1)
# print(a + b)
# c = np.arange(4).reshape(2, 2)
# print(a + c)


a2 = np.ones((3, 1, 5, 2))
b2 = np.ones((2, 5, 4))
c2 = np.ones((2, 5, 1))
print(a2 + c2)
print(a2 + b2)