import numpy as np
import time


def mean_value(X):
    X_mean = 0
    for i in range(len(X)):
        X_mean += X[i, :]
    X_mean = X_mean / len(X)
    return X_mean


def covariance(X, X_mean):
    n, p = X.shape
    cov = np.zeros((p, p))
    for i in range(n):
        cov += np.dot((X[i:i + 1, :] - X_mean).T, (X[i:i + 1, :] - X_mean))
    cov = cov / n
    return cov


def gaussian_probability(X, x):
    n, p = X.shape
    X_mean = mean_value(X)
    X_cov = covariance(X, X_mean)
    X_cov_det = np.linalg.det(X_cov)
    X_cov_inv = np.linalg.inv(X_cov)
    one = 1 / ((2 * np.pi) ** (p / 2))
    two = 1 / (X_cov_det ** (1 / 2))
    three = np.exp((-1 / 2) * (x - X_mean) @ X_cov_inv @ (x - X_mean).T)
    X_gaussian = one * two * three
    return X_gaussian


def decision():
    X_good_gaussian = gaussian_probability(X_good, x)
    X_bad_gaussian = gaussian_probability(X_bad, x)
    good = p_good * X_good_gaussian
    bad = p_good * X_bad_gaussian
    if good >= bad:
        print("密度为{}， 含糖量为{}的瓜，高斯贝叶斯预测为好瓜".format(x[0], x[1]))
    else:
        print("密度为{}， 含糖量为{}的瓜，高斯贝叶斯预测为坏瓜".format(x[0], x[1]))

start = time.time()
data = np.array([[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1],
                 [0.556, 0.215, 1], [0.403, 0.237, 1], [0.481, 0.149, 1], [0.437, 0.211, 1],
                 [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
                 [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0],
                 [0.719, 0.103, 0]])
X_good = np.array([i[0:2] for i in data if i[2] == 1])
X_bad = np.array([i[0:2] for i in data if i[2] == 0])
p_good = (len(X_good) + 1) / (len(data) + 2)  # 拉普拉斯修正
p_bad = (len(X_bad) + 1) / (len(data) + 2)  # 拉普拉斯修正
x = np.array([0.5, 0.3])

decision()
end = time.time()
print("高斯贝叶斯运行时间的一千倍为：", (end - start) * 1e3)