# Created by: Daniela Chanci
# Description: Application of the unsupervised clustering algorithm k-means
# to the public dataset "Epileptic Seizure Recognition Data Set" downloaded
# from the website data.world. Only two columns were used.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Load 2 columns the Epileptic Seizure Recognition Data Set
data_original = pd.read_csv("data.csv")
data = data_original.iloc[:,[16,55]]

# Apply k-means algorithm
k_means = KMeans(n_clusters = 5).fit(data)
y_kmeans = k_means.predict(data)
labels = np.transpose(k_means.labels_)
centers = k_means.cluster_centers_

# Plot clustered data 2D
fig1 = plt.figure()
plt.scatter(data['X16'], data['X55'], c=y_kmeans, s=10, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=5, alpha=0.8)
plt.title("K-means Clustering (5 Clusters)", fontsize=16)
plt.xlabel("X1", fontsize=13)
plt.ylabel("X2", fontsize=13)
plt.show()

