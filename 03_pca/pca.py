import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sample dataset
data = {
    "Math": [80, 90, 70, 60, 85],
    "Science": [85, 95, 75, 65, 88],
    "English": [78, 92, 72, 62, 84],
    "Computer": [90, 96, 80, 68, 91]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create PCA dataframe
pca_df = pd.DataFrame(
    pca_data,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nDataset after PCA:")
print(pca_df)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(sum(pca.explained_variance_ratio_))