import numpy as np
import matplotlib.pyplot as plt


def do_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]


def covariance(x, y):
    n = len(x)
    return np.dot(do_mean(x), do_mean(y)) / (n-1)

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

print("Covariance of the above data : {}".format(covariance(pageSpeeds, purchaseAmount)))


# creating a relationship between pageSpeed & purchaseAmount

purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

print("Covariance of the above data : {}".format(covariance(pageSpeeds, purchaseAmount)))


def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x, y) / stddevx / stddevy


print("Correlation of the above data : {}".format(correlation(pageSpeeds, purchaseAmount)))
print("Correlation of the above data- calculated via numpy : {}".format(np.corrcoef(pageSpeeds, purchaseAmount)))


