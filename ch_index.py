import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

# Load the data from the CSV file
data = pd.read_csv("mfcc_data_pca.csv")

# Separate the features (PCA components) from the labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Calculate the CH index for k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=8)
kmeans.fit(X)

# Calculate the within-cluster sum of squares (WSS) for each cluster
WSS = []
for label in range(3):
    WSS.append(np.sum((X[kmeans.labels_ == label] - kmeans.cluster_centers_[label])**2))

# Calculate the between-cluster sum of squares (BSS)
BSS = np.sum((kmeans.cluster_centers_ - np.mean(X, axis=0))**2) * (len(X) - len(np.unique(y))) / (len(np.unique(y)) - 1)

# Calculate the CH index
CH = (BSS / (len(np.unique(y)) - 1)) / (np.sum(WSS) / (len(X) - len(np.unique(y))))

print("CH index:", CH)
