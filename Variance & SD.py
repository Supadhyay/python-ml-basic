import numpy as np
import matplotlib.pyplot as plt
import math

incomes = np.random.normal(100, 20, 10000)

plt.hist(incomes, 50)
plt.show()

print("Standard Deviation : {}".format(incomes.std()))

print("Standard Deviation^2 : {}".format(math.pow(incomes.std(), 2)))

print("Variance : {}".format(incomes.var()))

