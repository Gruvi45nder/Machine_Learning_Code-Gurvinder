import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(40)
x = np.random.rand(100, 2)

#create the clusters
x[:50] += 1
x[50:] += 2

#Apply KMeans
Kmeans = KMeans(n_clusters=2)
Kmeans.fit(x)

#get cluster labels and centroids
labels = Kmeans.labels_
centroids = Kmeans.cluster_centers_

#plot the results
plt.figure(figsize=(8, 6))

# Plot the data points and color them according to their cluster label
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', marker='o', label="Data Points")

# Plot the centroids of the clusters
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label="Centroids")

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()