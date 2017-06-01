from sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split
import pylab as plt
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=1)
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

reg.fit(X_train, y_train)

print("Test set prediction : {}".format(reg.predict(X_test)))
print("Test set R * 2 : {:.2F}".format(reg.score(X_test, y_test)))

plt.plot(X, y, '^', markersize=8)
plt.plot(line, reg.predict(line))
plt.show()
