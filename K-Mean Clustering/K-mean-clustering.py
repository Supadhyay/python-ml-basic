import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Creating Random Cluster Data

def createClusteredData(N, K):
    np.random.seed(10)
    pointsPerCluster = float(N)/K
    X = []
    for i in range(K):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid= np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
    X = np.array(X)
    return X

data = createClusteredData(100, 5)

# Initiating and fitting the model
model = KMeans(n_clusters=3)

model = model.fit(scale(data))
print(model.labels_)


plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=model.labels_.astype(np.float))
plt.show()
