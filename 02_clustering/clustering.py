import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Student dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Exam_Score": [35, 40, 45, 50, 55, 60, 68, 72, 80, 88]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Select features for clustering
X = df[["Hours_Studied", "Exam_Score"]]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X)

print("\nClustered Dataset:")
print(df)

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot the clusters
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Hours_Studied",
    y="Exam_Score",
    hue="Cluster",
    palette="deep",
    s=100
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    color="black",
    label="Centroids"
)

plt.title("K-Means Clustering of Students")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)

plt.show()