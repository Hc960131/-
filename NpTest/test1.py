import numpy as np

a = np.arange(32).reshape(4, 2, 4)
b = np.arange(2).reshape(2, 1)
print(a + b)
c = np.arange(4).reshape(2, 2)
print(a + c)
