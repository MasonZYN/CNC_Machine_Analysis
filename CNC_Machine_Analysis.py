import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Update these paths with the correct paths to your CSV files on your computer
file_paths = [
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\1.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\2.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\4.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\6.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\9.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\13.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\14.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\15.csv',
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\18.csv',  # Renamed from 96INTVL.csv
    r'C:\Users\Mohsen\OneDrive\Desktop\PhD\Kuts (Modelling from Measurement) Course\Final Project\19.csv'   # Renamed from 96INTVL.csv
]

# Load all the CSV files into a list of DataFrames
dataframes = [pd.read_csv(file) for file in file_paths]

# Combine the data from all CSV files
combined_data = pd.concat(dataframes, ignore_index=True)

# Assuming the columns are 'MaxIrms1' (current), 'MaxUrms1' (voltage), 'MaxP1' (power)
# Convert these columns to numeric, forcing non-numeric values like 'OVER' to NaN
combined_data['MaxIrms1'] = pd.to_numeric(combined_data['MaxIrms1'], errors='coerce')
combined_data['MaxUrms1'] = pd.to_numeric(combined_data['MaxUrms1'], errors='coerce')
combined_data['MaxP1'] = pd.to_numeric(combined_data['MaxP1'], errors='coerce')

# Drop rows with NaN values (since they likely contain non-numeric values like 'OVER')
combined_data.dropna(subset=['MaxIrms1', 'MaxUrms1', 'MaxP1'], inplace=True)

# Extract the relevant columns for clustering
X = combined_data[['MaxIrms1', 'MaxUrms1', 'MaxP1']]

# Normalize the data to bring all values to a comparable scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA-transformed data
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', s=50)
plt.title('PCA-Transformed Data')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Use the elbow method to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# After inspecting the plot, choose the optimal number of clusters (let's say 3)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
combined_data['Cluster'] = clusters

# Visualize the clusters using PCA
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('Clusters of CNC Operational Modes')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Analyze the clusters and find the cluster with the lowest power consumption
print("Cluster Averages:\n", combined_data.groupby('Cluster').mean())
optimal_cluster = combined_data.groupby('Cluster')['MaxP1'].mean().idxmin()
print(f"\nThe optimal cluster for power consumption is: {optimal_cluster}")

# Extract optimal settings
optimal_settings = combined_data[combined_data['Cluster'] == optimal_cluster]
print("Optimal settings for reduced power consumption:")
print(optimal_settings)
