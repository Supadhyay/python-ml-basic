import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Creating Random Cluster Data

def createClusteredData(N, K):
    np.random.seed(6)
    pointsPerCluster = float(N)/K
    X = []
    Y = []
    for i in range(K):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid= np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            Y.append(i)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

(X, Y) = createClusteredData(100, 3)


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y.astype(np.float))
plt.show()


C = 1.0

svc = svm.SVC(kernel='linear', C=C).fit(X, Y)


def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                         np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y.astype(np.float))
    plt.show()


plotPredictions(svc)

print(svc.predict([[200000, 40]]))
