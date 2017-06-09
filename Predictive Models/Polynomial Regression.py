import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
pageSpeed = np.random.normal(3.0, 1.0, 1000)

# Linear Relationship - r**2 is closer to 1
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeed

plt.scatter(pageSpeed, purchaseAmount)
plt.show()

# fitting the poly regression to the curve

x = np.array(pageSpeed)
y = np.array(purchaseAmount)


p4 = np.poly1d(np.polyfit(x, y, 4))

xp = np.linspace(0, 7, 100)
plt.scatter(pageSpeed, purchaseAmount)
plt.plot(xp, p4(xp), c='r')
plt.show()

r2 = r2_score(y, p4(x))


print("R2 Score : {}".format(r2))
