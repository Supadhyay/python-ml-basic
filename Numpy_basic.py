import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Adding an outlier to check the effect on median vs mean
# incomes = np.append(incomes, [10000000000])


print("Dealing with incomes")

incomes = np.random.normal(27000, 15000, 10000)

print("Mean : {}".format(np.mean(incomes)))
print("Median : {}".format(np.median(incomes)))

plt.hist(incomes, 500)
plt.show()


print("Dealing with ages")

ages = np.random.randint(18, high=90, size=500)
print("Mean : {}".format(np.mean(ages)))
print("Median : {}".format(np.median(ages)))
print("Mode : {}".format(stats.mode(ages)))


