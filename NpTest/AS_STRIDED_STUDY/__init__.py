# as_stride主要有两个作用：
# 第一个作用是使用一个更小内存的方案，生成序列化数据
# 第二个作用是更快的对一系列数据左平滑，求平均值，最大值等处理
# 还有一个额外作用，就是可以执行卷积操作
# test1完成第一个作用的学习
# test2完成第二个作用的学习
# test3完成执行卷积操作

import numpy as np


def public_check(sequence, shape, stride):
    needed = np.sum((np.array(shape) - 1) * np.array(stride))
    if needed >= sequence.nbytes:
        raise ValueError("数组越界")