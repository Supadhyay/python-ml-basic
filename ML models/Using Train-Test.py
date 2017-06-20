import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
polyFitValue = 5

pageSpeed = np.random.normal(3.0, 1.0, 100)

# Linear Relationship - r**2 is closer to 1
purchaseAmount = np.random.normal(50.0, 10.0, 100) / pageSpeed

plt.scatter(pageSpeed, purchaseAmount)
plt.suptitle("Total DataSet")
plt.show()


# Training DatSet
trainX = pageSpeed[:80]
testX = pageSpeed[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

plt.scatter(trainX, trainY)
plt.suptitle("Training Data")
plt.show()


# Testing DatSet

plt.scatter(testX, testY)
plt.suptitle("Testing Data")
plt.show()


# poly regression

x = np.array(trainX)
y = np.array(trainY)

# Degree of polynomial here could change the over-fitting / under-fitting the data
p4 = np.poly1d(np.polyfit(x, y, polyFitValue))

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])

plt.scatter(x, y)
plt.suptitle("Manual Poly Regression Train Data-Set")
plt.plot(xp, p4(xp), c='r')
plt.show()


x = np.array(testX)
y = np.array(testY)

p4 = np.poly1d(np.polyfit(x, y, polyFitValue))

xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])

plt.scatter(x, y)
plt.suptitle("Manual Poly Regression Test Data-Set")
plt.plot(xp, p4(xp), c='r')
plt.show()

r2_test = r2_score(testY, p4(testX))
r2_train = r2_score(trainY, p4(trainX))

print("Train r2 score : {}".format(r2_train))
print("Test r2 score : {}".format(r2_test))
