#KNN is basically depends on the distance
# There are 2 ways for calculating the distance 

import numpy as np
from collections import Counter

def euclideanDist(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def manhattanDist(p1, p2):
    return np.sum(np.abs(p1 - p2))

def predict_knn(X_train, y_train, new_point, k, distance_metric=euclideanDist):
    # 1. Calculate distances from the new point to all training points
    distances = [distance_metric(new_point, x) for x in X_train]

    # 2. Find the k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]

    # 3. Get the labels of those neighbors
    k_nearest_labels = [y_train[i] for i in k_nearest_indices]

    # 4. Vote!
    most_common = Counter(k_nearest_labels).most_common(1)

    return most_common[0][0]

X_train = np.array([[1, 2], [2, 3], [3, 4], [8, 7]])
y_train = ['A', 'A', 'B', 'B']
new_point = np.array([2, 2])

print(predict_knn(X_train, y_train, new_point, k=3))  # Output: A

