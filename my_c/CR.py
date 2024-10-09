#计算矩阵特征值
import numpy as np
from numpy import linalg as LA

A = np.array([[1,2, 3, 3],[1/2, 1, 2, 2],[1/3, 1/2, 1, 1/2],[1/3, 1/2, 2, 1]])

column_sums = A.sum(axis=0)

# 将矩阵的每个元素除以其所在列的和，进行归一化
normalized_A = A / column_sums

# 计算权重：对归一化后的矩阵的每一行求平均值
weights = normalized_A.mean(axis=1)

# 输出归一化后的矩阵和计算得到的权重
print("归一化后的判断矩阵:\n", normalized_A)
print("权重:", weights)