import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("\nFirst 5 Rows:")
print(df.head())

# Annual Income and Spending Score
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
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("images/elbow_method.png")
plt.show()

# Train Final K-Means Model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Customer Segments Visualization
plt.figure(figsize=(8, 6))

for i in range(5):
    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=100,
        label=f'Cluster {i + 1}'
    )

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
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

# User Input Section
print("\n--- Customer Cluster Predictor ---")

annual_income = float(input("Enter Annual Income (k$): "))
spending_score = float(input("Enter Spending Score (1-100): "))

new_customer = [[annual_income, spending_score]]

cluster = kmeans.predict(new_customer)

print(f"\nCustomer belongs to Cluster {cluster[0] + 1}")

print("\nClustering Completed Successfully!")