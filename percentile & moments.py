import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

vals = np.random.normal(0, .5, 10000)

plt.hist(vals, 50)
plt.show()

print("50th Percentile : {}".format(np.percentile(vals, 50)))

print("60th Percentile : {}".format(np.percentile(vals, 60)))

print("99th Percentile : {}".format(np.percentile(vals, 99)))


# Moments of data

print("First Moment ~ Mean : {}".format(np.mean(vals)))

print("Second Moment ~ Variance : {}".format(np.var(vals)))

print("Third Moment ~ Skew : {}".format(sp.skew(vals)))

print("Fourth Moments ~ kurtosis : {}".format(sp.kurtosis(vals)))
