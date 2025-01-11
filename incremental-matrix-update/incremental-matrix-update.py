import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import svds
from time import time

# Generate a sparse matrix using real-world-like data
def generate_sparse_data(rows, cols, density=0.05):
    rng = np.random.default_rng(42)
    data = rng.choice([0, 1], size=(rows, cols), p=[1 - density, density]).astype(float)
    return csr_matrix(data)

# Incremental update to a sparse matrix
def incremental_update(sparse_matrix, new_rows=None, new_cols=None):
    if new_rows is not None:
        sparse_matrix = vstack([sparse_matrix, csr_matrix(new_rows.astype(float))])
    if new_cols is not None:
        sparse_matrix = hstack([sparse_matrix, csr_matrix(new_cols.astype(float))])
    return csr_matrix(sparse_matrix)

# Measure performance of SVD with incremental updates
def measure_performance(sparse_matrix, new_rows):
    print("Original matrix shape:", sparse_matrix.shape)

    # Perform initial SVD
    start_time = time()
    u, s, vt = svds(sparse_matrix, k=10)
    initial_time = time() - start_time
    print(f"Initial SVD time: {initial_time:.4f} seconds")

    # Incrementally update the matrix
    sparse_matrix = incremental_update(sparse_matrix, new_rows=new_rows)
    print("Updated matrix shape:", sparse_matrix.shape)

    # Perform SVD on the updated matrix
    start_time = time()
    u_updated, s_updated, vt_updated = svds(sparse_matrix, k=10)
    updated_time = time() - start_time
    print(f"Updated SVD time: {updated_time:.4f} seconds")

    return initial_time, updated_time

# Main script
if __name__ == "__main__":
    # Generate a sparse matrix (real-world-like data)
    sparse_matrix = generate_sparse_data(rows=1000, cols=500, density=0.01)

    # New rows to add incrementally
    new_rows = np.random.choice([0, 1], size=(100, 500), p=[0.99, 0.01]).astype(float)

    # Measure performance
    initial_time, updated_time = measure_performance(sparse_matrix, new_rows)

    # Display results
    print("\nPerformance Comparison:")
    print(f"Initial SVD Time: {initial_time:.4f} seconds")
    print(f"Incremental SVD Time: {updated_time:.4f} seconds")

    # Plot the performance comparison
    labels = ['Initial SVD', 'Incremental SVD']
    times = [initial_time, updated_time]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=['blue', 'orange'], alpha=0.7)
    plt.title('Performance Comparison: SVD vs Incremental SVD')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Method')
    plt.show()


