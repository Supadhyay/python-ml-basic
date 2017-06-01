import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


x = np.arange(-3, 3, .001)

plt.plot(x, norm.pdf(x))
plt.show()

# multiple graphs in one

x = np.arange(-3, 3, .001)

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, .5))

# Change the path below to your desktop path
# plt.savefig('C:\\Users\\supadhyay\\Desktop\\plot.png', format='png')
plt.show()


# Adding a grid

axes = plt.axes()
axes.grid()

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, .5))

plt.show()

# Changing the color and types of lines and adding lables


axes = plt.axes()
axes.grid()

plt.xlabel('Random Data')
plt.ylabel('Probability')

plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, .5), 'r:')

plt.legend(['RD1', 'RD2'], loc=4)
plt.show()

# Scatter plot

x = np.random.randn(500)
y = np.random.randn(500)
plt.scatter(x, y)
plt.show()
