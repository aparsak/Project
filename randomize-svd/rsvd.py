import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import load_digits
from time import time

# Load real-world data (digits dataset)
data = load_digits()
X = data.data  # shape (1797, 64)

# Function to measure reconstruction error
def reconstruction_error(original, reconstructed):
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)

# Function to perform standard SVD
def perform_standard_svd(X, n_components):
    svd = TruncatedSVD(n_components=n_components)
    start_time = time()
    X_reduced = svd.fit_transform(X)
    X_reconstructed = svd.inverse_transform(X_reduced)
    elapsed_time = time() - start_time
    error = reconstruction_error(X, X_reconstructed)
    return elapsed_time, error

# Function to perform randomized SVD
def perform_randomized_svd(X, n_components):
    start_time = time()
    U, Sigma, VT = randomized_svd(X, n_components=n_components, random_state=42)
    X_reduced = np.dot(U, np.diag(Sigma))
    X_reconstructed = np.dot(X_reduced, VT)
    elapsed_time = time() - start_time
    error = reconstruction_error(X, X_reconstructed)
    return elapsed_time, error

# Number of components for dimensionality reduction
n_components = 20

# Standard SVD
standard_svd_time, standard_svd_error = perform_standard_svd(X, n_components)

# Randomized SVD
randomized_svd_time, randomized_svd_error = perform_randomized_svd(X, n_components)

# Display results
print("Standard SVD:")
print(f"Time: {standard_svd_time:.4f} seconds")
print(f"Reconstruction Error: {standard_svd_error:.6f}")

print("\nRandomized SVD:")
print(f"Time: {randomized_svd_time:.4f} seconds")
print(f"Reconstruction Error: {randomized_svd_error:.6f}")

# Bar plot for comparison
labels = ['Standard SVD', 'Randomized SVD']
time_values = [standard_svd_time, randomized_svd_time]
error_values = [standard_svd_error, randomized_svd_error]

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot execution time
ax1.bar(labels, time_values, color=['blue', 'orange'], alpha=0.7)
ax1.set_ylabel('Time (seconds)', color='black')
ax1.set_title('Comparison of Standard SVD and Randomized SVD')

# Plot reconstruction error on secondary axis
ax2 = ax1.twinx()
ax2.plot(labels, error_values, color='red', marker='o', linestyle='--', label='Reconstruction Error')
ax2.set_ylabel('Reconstruction Error', color='red')

plt.legend(loc='upper right')
plt.show()
