import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Parameters
num_points = 10000  # Number of points to sample
dimensions = [2, 1000, 10000]  # Different dimensions to sample

# Initialize the figure for multiple plots
plt.figure(figsize=(15, 5))

for i, dim in enumerate(dimensions, 1):
    # Sample points from a Gaussian distribution
    points = np.random.normal(loc=0, scale=1, size=(num_points, dim))
    
    # Compute pairwise Euclidean distances
    distances = distance.pdist(points, 'euclidean')
    flat_distances = distances.flatten()
    
    # Plot the Euclidean distance distribution for the current dimension
    plt.subplot(1, 3, i)
    plt.hist(flat_distances, bins=50, density=True, alpha=0.7, color='blue')
    plt.title(f'Euclidean Distance Distribution (d={dim})')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.grid()

# Save the figure with all subplots
plt.tight_layout()
plt.savefig('images/euclidean_distance_distributions.png')
plt.close()
print("Euclidean distance distribution plots saved as 'euclidean_distance_distributions.png'")
