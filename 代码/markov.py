import numpy as np

# 转移矩阵
A = np.array([[0.8, 0.2], [0.5, 0.5]])

res = A[1] @ A @ A @ A

print("射中的概率为%.4f， 射不中的概率为%.4f" % (res[0], res[1]))

