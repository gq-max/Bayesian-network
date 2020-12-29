import numpy as np
import time


def variance(x, mean_value):
    var = 0
    for i in x:
        var += (i - mean_value) ** 2
    var = np.sqrt((var / len(x)))
    return var


def conditional_probability():
    c = np.sqrt(2 * np.pi)
    p_density_good = ((1 / (c * var_density_good)) *
                      np.exp(-(x[0] - mean_density_good) ** 2 / var_density_good ** 2))
    p_density_bad = ((1 / (c * var_density_bad)) *
                     np.exp(-(x[0] - mean_density_bad) ** 2 / var_density_bad ** 2))
    p_sugar_good = ((1 / (c * var_sugar_good)) *
                    np.exp(-(x[0] - mean_sugar_good) ** 2 / var_sugar_good ** 2))
    p_sugar_bad = ((1 / (c * var_sugar_bad)) *
                   np.exp(-(x[0] - mean_sugar_bad) ** 2 / var_sugar_bad ** 2))
    return p_density_good, p_density_bad, p_sugar_good, p_sugar_bad


def decision():
    p_density_good, p_density_bad, p_sugar_good, p_sugar_bad = conditional_probability()
    good = p_density_good * p_sugar_good * p_good
    bad = p_density_bad * p_sugar_bad * p_bad
    if good >= bad:
        print("密度为{}， 含糖量为{}的瓜，朴素高斯贝叶斯预测为好瓜".format(x[0], x[1]))
    else:
        print("密度为{}， 含糖量为{}的瓜，朴素高斯贝叶斯预测为坏瓜".format(x[0], x[1]))


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
mean_density_good = np.mean(X_good[:, 0])
mean_density_bad = np.mean(X_bad[:, 0])
mean_sugar_good = np.mean(X_good[:, 1])
mean_sugar_bad = np.mean(X_bad[:, 1])
var_density_good = variance(X_good[:, 0], mean_density_good)
var_density_bad = variance(X_bad[:, 0], mean_density_bad)
var_sugar_good = variance(X_good[:, 1], mean_sugar_good)
var_sugar_bad = variance(X_bad[:, 1], mean_sugar_bad)
end = time.time()
decision()
print("朴素高斯贝叶斯运行时间的一千倍为：", (end - start) * 1e3)
