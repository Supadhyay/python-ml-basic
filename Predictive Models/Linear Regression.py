import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


pageSpeed = np.random.normal(3.0, 1.0, 1000)

# Linear Relationship - r**2 is closer to 1
purchaseAmount = 100 - (pageSpeed + np.random.normal(0, 0.1, 1000)) * 3

# Non Linear Relationship - r**2 is further from 1
# purchaseAmount = pageSpeed * pageSpeed * pageSpeed

plt.scatter(pageSpeed, purchaseAmount)
plt.show()


slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeed, purchaseAmount)

print("slope : {}".format(slope), "intercept: {}".format(intercept), "r_value: {}".format(r_value),
      "r_value^ 2: {}".format(r_value ** 2), "p_value: {}".format(p_value), "std_err: {}".format(std_err))

# using slope and intercept to predict the values vs observed


def predict(x):
    return slope * x + intercept


fitLine = predict(pageSpeed)

plt.scatter(pageSpeed, purchaseAmount)
plt.plot(pageSpeed, fitLine, c='r')
plt.show()
