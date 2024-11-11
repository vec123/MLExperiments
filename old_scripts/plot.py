import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# 1. White Noise
white_noise = np.random.normal(0, 1, size=(100, 2))

# 2. Data 
np.random.seed(42) 
x = np.linspace(-3* np.pi, 3 * np.pi, 5000)  # x values

mean_x =   x
variance_x = 0.5
x_samples = np.random.normal(loc=mean_x, scale=np.sqrt(variance_x), size=len(x))

mean_y = np.sin(x)
variance_y = 0.5#+0.5*np.sin(x)
y_samples = np.random.normal(loc=mean_y, scale=np.sqrt(variance_y), size=len(x))

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x_samples, y_samples, alpha=0.7)
#plt.scatter(x, y_samples, color='red', label='Sampled Data (y)', alpha=0.7, s=10)
#plt.scatter(x, mean_y + np.sqrt(variance_y), mean_y - np.sqrt(variance_y), color='gray', alpha=0.3, label='Variance Region')
plt.title('Distribution with sinusoidal mean and constant variance')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig("Sinusoidal Mean and constant variance.png")


# Parameters for the circle
radius = 5
t = np.linspace(0, 2 * np.pi, 5000)  # Parameter for the circle

# Parametric equations for the circle
mean_x = radius * np.cos(t)
mean_y = radius * np.sin(t)

variance_x = 0.5
x_samples = np.random.normal(loc=mean_x, scale=np.sqrt(variance_x), size=len(x))

variance_y = 0.5#+0.5*np.sin(x)
y_samples = np.random.normal(loc=mean_y, scale=np.sqrt(variance_y), size=len(x))
# Plot
plt.figure(figsize=(12, 6))
plt.scatter(x_samples, y_samples, alpha=0.7)
#plt.scatter(x, y_samples, color='red', label='Sampled Data (y)', alpha=0.7, s=10)
#plt.scatter(x, mean_y + np.sqrt(variance_y), mean_y - np.sqrt(variance_y), color='gray', alpha=0.3, label='Variance Region')
plt.title('Distribution with circular mean and constant variance')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig("Circular Mean and constant variance.png")