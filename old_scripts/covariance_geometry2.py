import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

# 1. White Noise (Identity Covariance Matrix)
mean = [0, 0]
cov_matrix_white_noise = [[1, 0], [0, 1]]
data_white_noise = np.random.multivariate_normal(mean, cov_matrix_white_noise, size=500)

# 2. Data with a Specific Covariance Matrix
cov_matrix_specific = [[3, 1], [1, 2]]
data_specific_cov = np.random.multivariate_normal(mean, cov_matrix_specific, size=500)

# 3. Data Distributed Around a Parabola with Covariance Noise
x_values_parabola = np.linspace(-3, 3, 500)
parabola_values = x_values_parabola**2
data_parabola = np.vstack((x_values_parabola, parabola_values)).T

# Add noise with a covariance matrix to the parabola
cov_matrix_parabola_noise = [[0.5, 0.2], [0.2, 0.3]]  # Covariance matrix for noise
noise_parabola = np.random.multivariate_normal([0, 0], cov_matrix_parabola_noise, size=500)
data_parabola += noise_parabola

# 4. Data Distributed Around a Circle with Covariance Noise
number_of_points = 5000
theta = np.linspace(0, 2 * np.pi, number_of_points)
radius = 5
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
data_circle = np.vstack((circle_x, circle_y)).T
data_circle_noise = data_circle
# Add noise with a covariance matrix to the circle
cov_matrix_circle_noise = [[1, 0.1], [1, 0.1]]  # Covariance matrix for noise
noise_circle = np.random.multivariate_normal([0, 0], cov_matrix_circle_noise, size=number_of_points)
data_circle_noise += noise_circle


# 6. Data Distributed Along a Diagonal Line
cov_matrix_diagonal = [[1, 0.9], [0.9, 1]]
data_diagonal = np.random.multivariate_normal(mean, cov_matrix_diagonal, size=500)


# 7. Data Distributed Along a y axis 
# Generate uniform data along the y-axis
y_values = np.random.uniform(-100, 100, 100)
x_values = np.random.normal(0, np.sqrt(5), 100)  # Standard deviation is the square root of variance (sqrt(5))
#x_values = x_values *0
data_uniform_line = np.vstack((x_values, y_values)).T

# 8. Data Distributed Along a digaonal rotated  
phi = np.pi / 4  # 45 degrees
transformation_matrix = np.array([
    [np.cos(phi), np.sin(phi)],
    [-np.sin(phi), np.cos(phi)]
])
data_uniform_line_tf = data_uniform_line @ transformation_matrix.T
original_cov_matrix = np.cov(data_uniform_line, rowvar=False)
transformed_cov_matrix = transformation_matrix @ original_cov_matrix @ transformation_matrix.T

print("Original Covariance Matrix:", original_cov_matrix)
print("TF Covariance Matrix:",transformed_cov_matrix )



# 9. Data Distributed Along a y axis 
y_values = np.random.uniform(-100, 100, 500)
y_values.sort() 
mid_point = len(y_values) // 2
y_values_first_half = y_values[:mid_point]
y_values_second_half = y_values[mid_point:]
cov_matrix_first_half = [[5, 0], [0, 0.1]]  # Covariance matrix for the first half
noise_first_half = np.random.multivariate_normal([0, 0], cov_matrix_first_half, size=mid_point)
cov_matrix_second_half = [[100, 0], [0, 0.1]]  # Covariance matrix for the second half
noise_second_half = np.random.multivariate_normal([0, 0], cov_matrix_second_half, size=len(y_values) - mid_point)
x_values_first_half = noise_first_half[:, 0]
x_values_second_half = noise_second_half[:, 0]
x_values = np.concatenate([x_values_first_half, x_values_second_half])
y_values = np.concatenate([y_values_first_half, y_values_second_half])
data_combined = np.vstack((x_values, y_values)).T

# 10. Data Distributed Along a y axis continous
# Generate y-values uniformly along the y-axis
y_values = np.random.uniform(-100, 100, 5000)
y_values.sort()  
# Initialize arrays to store x-values
x_values = np.zeros_like(y_values)
noise_values = []
for i, y in enumerate(y_values):
    # Define a continuously varying covariance for x based on y
    #variance_x = 5 + (y + 100) * 0.95  # Map y in [-100, 100] to a variance in [5, 195]
    variance_x = 5 + 300 * (0.5 + 0.5 * np.sin(3*y * np.pi /50))  # Scales sine wave to [5, 50]
    covariance_matrix = [[variance_x, 0], [0, 0.1]]  # Keep variance for y small
    noise = np.random.multivariate_normal([0, 0], covariance_matrix)
    noise_values.append(noise)
    x_values[i] = noise[0]
    
data_continuous_covariance = np.vstack((x_values, y_values)).T

# 11. Map to circle
theta = (y_values - y_values.min()) / (y_values.max() - y_values.min()) * 2 * np.pi  # Map y in [-100, 100] to [0, 2*pi]
print("len theta:", len(theta))
radius = 200
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
noise_circle = np.zeros((len(theta), 2))
for i in range(len(theta)):
    # Determine the normal direction at this point (perpendicular to the tangent)
    normal_direction = np.array([np.cos(theta[i]), np.sin(theta[i])])
    noise_circle[i] = noise_values[i][0] * normal_direction 
data_circle_transformed = np.vstack((circle_x, circle_y)).T + noise_circle


# Map to a parabola
t = (y_values - y_values.min()) / (y_values.max() - y_values.min()) * 2 - 1  # Map y to range [-1, 1]
a = 1 # Controls the width of the parabola
parabola_x = t
parabola_y = a * t**2
noise_parabola = np.zeros((len(t), 2))
for i in range(len(t)):
    # The tangent vector at this point is [1, 2*a*t[i]]
    tangent = np.array([1, 2 * a * t[i]])
    # The normal vector is perpendicular to the tangent
    normal_direction = np.array([-tangent[1], tangent[0]])
    normal_direction /= np.linalg.norm(normal_direction)  # Normalize
    noise_parabola[i] = noise_values[i][0] * normal_direction
data_parabola_transformed = np.vstack((parabola_x, parabola_y)).T +noise_parabola*0.01



# Plot settings
number_of_figures = 8
fig_size = (8, 8)

# Plot 1: White Noise
plt.figure(figsize=fig_size)
plt.scatter(data_white_noise[:, 0], data_white_noise[:, 1], alpha=0.6, edgecolor='k')
plt.title('White Noise (Identity Covariance)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_1_white_noise.png')
plt.close()

# Plot 2: Specific Covariance
plt.figure(figsize=fig_size)
plt.scatter(data_specific_cov[:, 0], data_specific_cov[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with Specific Covariance')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_2_specific_covariance.png')
plt.close()

# Plot 3: Data Around a Parabola
plt.figure(figsize=fig_size)
plt.scatter(data_parabola[:, 0], data_parabola[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Parabola')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_3_parabola.png')
plt.close()

# Plot 4: Data Around a Circle
plt.figure(figsize=fig_size)
plt.scatter(data_circle_noise[:, 0], data_circle_noise[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_4_circle.png')
plt.close()

# Plot 5: Data Around a Diagonal Line
plt.figure(figsize=fig_size)
plt.scatter(data_diagonal[:, 0], data_diagonal[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Around a Diagonal')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_5_diagonal.png')
plt.close()

# Plot 6: Data Uniform Line
plt.figure(figsize=fig_size)
plt.scatter(data_uniform_line[:, 0], data_uniform_line[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data Uniform Line (regular distribution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_6_uniform_line (regular distribution).png')
plt.close()

# Plot 7: Transformed Uniform Line
plt.figure(figsize=fig_size)
plt.scatter(data_uniform_line_tf[:, 0], data_uniform_line_tf[:, 1], alpha=0.6, edgecolor='k')
plt.title('Transformed Uniform Line  (regular distribution)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_7_transformed_uniform_line (regular distribution).png')
plt.close()

# Plot 8: Data with Two Variances
plt.figure(figsize=fig_size)
plt.scatter(data_combined[:, 0], data_combined[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with Two Variances')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_8_two_variances.png')
plt.close()

# Plot 8: Data with Two Variances
plt.figure(figsize=fig_size)
plt.scatter(data_continuous_covariance[:, 0], data_continuous_covariance[:, 1], alpha=0.6, edgecolor='k')
plt.title('Data with continous varying variances (sine)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('plot_9_continous_varying_variances_sine.png')
plt.close()


plt.figure(figsize=fig_size)
plt.scatter(data_circle_transformed[:, 0], data_circle_transformed[:, 1], alpha=0.6, edgecolor='k')
plt.title('Transformed data along a circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('transformed_circle_mean_sine_variance.png')
plt.close()


plt.figure(figsize=fig_size)
plt.scatter(data_parabola_transformed[:, 0], data_parabola_transformed[:, 1], alpha=0.6, edgecolor='k')
plt.title('Transformed data along a parabola')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('transformed_parabola_mean_sine_variance.png')
plt.close()

