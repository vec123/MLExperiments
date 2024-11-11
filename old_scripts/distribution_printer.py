import numpy as np
import matplotlib.pyplot as plt

# Generate x values for plotting
x = np.linspace(-3, 3, 1000)

# Isotropic Gaussian: zero mean, variance 1
mean1, var1 = 0, 1
gaussian1 = (1 / np.sqrt(2 * np.pi * var1)) * np.exp(-0.5 * ((x - mean1) ** 2) / var1)

# Two Gaussians: mean 0.8, variance 0.5
mean2, var2 = 1.5, 0.5
gaussian2 = (1 / np.sqrt(2 * np.pi * var2)) * np.exp(-0.5 * ((x - mean2) ** 2) / var2)

mean3, var3 = -1.5, 0.5
gaussian3 = (1 / np.sqrt(2 * np.pi * var3)) * np.exp(-0.5 * ((x - mean3) ** 2) / var3)

# Create a larger figure
plt.figure(figsize=(12, 8))  # Set the figure size to 12x8 inches

# Plotting
plt.plot(x, gaussian1, color='red', label='Isotropic Gaussian (mean=0, var=1)')
plt.plot(x, gaussian2, color='blue', linestyle='--', label='Gaussian 1 (mean=0.8, var=0.5)')
plt.plot(x, gaussian3, color='green', linestyle='-.', label='Gaussian 3 (mean=-0.8, var=0.5)')

# Adding labels and legend
plt.title('Gaussian Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

# Save the plot as a PNG file
plt.grid()
plt.savefig('gaussian_plot.png', dpi=300, bbox_inches='tight')
plt.close()
