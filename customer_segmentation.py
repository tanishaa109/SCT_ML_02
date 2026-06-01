# customer_segmentation.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("\nFirst 5 Rows:")
print(df.head())

# Features for Clustering
X = df.iloc[:, [3, 4]].values

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("images/elbow_method.png")
plt.show()

# K-Means Model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(8,6))

plt.scatter(
    X[y_kmeans==0,0],
    X[y_kmeans==0,1],
    s=100,
    label='Cluster 1'
)

plt.scatter(
    X[y_kmeans==1,0],
    X[y_kmeans==1,1],
    s=100,
    label='Cluster 2'
)

plt.scatter(
    X[y_kmeans==2,0],
    X[y_kmeans==2,1],
    s=100,
    label='Cluster 3'
)

plt.scatter(
    X[y_kmeans==3,0],
    X[y_kmeans==3,1],
    s=100,
    label='Cluster 4'
)

plt.scatter(
    X[y_kmeans==4,0],
    X[y_kmeans==4,1],
    s=100,
    label='Cluster 5'
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker='X',
    label='Centroids'
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.savefig("images/customer_segments.png")
plt.show()

print("\nClustering Completed Successfully!")