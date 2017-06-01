import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom
from scipy.stats import poisson

# Uniform distribution
values = np.random.uniform(20, 100, 10000)
plt.hist(values, 50)
plt.show()

# Normal Distributions

x = np.arange(-3, 3, .001)

plt.plot(x, norm.pdf(x))
plt.show()

# mu is the mean
# sigma is the SD

mu = 5.0
sigma = 2.0
values = np.random.normal(mu, sigma, 10000)

plt.hist(values, 50)
plt.show()


# Exponential Distribution

x = np.arange(0, 10, .0001)

plt.plot(x, expon.pdf(x))
plt.show()


# # Probability mass function - discrete data

# Binomial PMF

n, p = 10, .5
x = np.arange(0, 10, .001)

plt.plot(x, binom.pmf(x, n, p))

plt.show()

# Poisson PMF -

mu = 500
x = np.arange(400, 600, .5)

plt.plot(x, poisson.pmf(x, mu))

plt.show()
