from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()


# Understanding the data set
print (' ############################################################ ')
print("Cancer Keys : {}".format(cancer.keys()))
print("Target : {}".format(cancer['target_names']))
print("Features : {}".format(cancer['feature_names']))
print("Shape : {}".format(cancer['data'].shape))
print ("Data Distribution : {}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

# Create a learning / training data set
print (' ############################################################ ')

X_train, X_test, y_train, y_test = train_test_split(cancer["data"], cancer["target"], random_state=0)

print("X train data : {}".format(X_train.shape))
print("X test data : {}".format(X_test.shape))


# Use KnNeibhours to fit then predict and test the prediction to product the model accuracy
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Model Accuracy : {}".format(np.mean(y_pred == y_test)))

