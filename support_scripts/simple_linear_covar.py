import sys
import os

import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(".."))
import my_lib

mean_2d = [0, 0]  # Mean for each dimension (centered at origin)
covariance_matrix = [[1, 0.5], [0.5, 1]]  # Covariance matrix for correlated 2D Gaussian
num_samples = 1000  # Number of samples to generate
# Generate samples from a 2D Gaussian distribution
X = np.random.multivariate_normal(mean_2d, covariance_matrix, num_samples)
# Separate into x and y for plotting
x_orig, y_orig = X[:, 0], X[:, 1]
X = np.array([x_orig, y_orig])
print(X.shape)
XXT = X@X.T/num_samples
eig_XXT = my_lib.get_eigenspace(XXT)

print("XXT: ", XXT)
print("Eigenspace of XXT: ", eig_XXT[0])

X = np.random.multivariate_normal(mean_2d, XXT, num_samples)
x, y = X[:, 0], X[:, 1]
# Plot the 2D Gaussian distribution as a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_orig, y_orig, alpha=0.5, marker='o', edgecolor='k', s=10)
plt.scatter(x, y, alpha=0.5, marker='o', edgecolor='k', s=10)
plt.title("2D Gaussian Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')
plt.savefig('../data_images/2d_gaussian_Sampled.png')